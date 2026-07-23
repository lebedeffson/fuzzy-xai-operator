from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from decimal import Decimal
from hashlib import sha256
from types import MappingProxyType
from typing import Mapping

from .models import Case

COST_ABS_TOL = Decimal("1e-12")
COST_REL_TOL = Decimal("1e-12")


def _decimal(value: object) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def _freeze(values: Mapping[str, object]) -> Mapping[str, Decimal]:
    return MappingProxyType({key: _decimal(value) for key, value in sorted(values.items())})


@dataclass(frozen=True)
class CostRegistry:
    atom_costs: Mapping[str, Decimal]
    action_costs: Mapping[str, Decimal]
    rollback_costs: Mapping[str, Decimal]
    human_approval_costs: Mapping[str, Decimal]
    fixed_costs: Mapping[str, Decimal]
    schema_version: str = "h10-c3-cost-registry-v1"
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "atom_costs",
            "action_costs",
            "rollback_costs",
            "human_approval_costs",
            "fixed_costs",
        ):
            object.__setattr__(self, name, _freeze(getattr(self, name)))
        payload = self.to_payload()
        digest = sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        object.__setattr__(self, "sha256", digest)

    @classmethod
    def from_case(cls, case: Case) -> CostRegistry:
        return cls(
            atom_costs={item.atom_id: item.cost for item in case.candidates},
            action_costs={item.atom_id: item.action_cost for item in case.candidates},
            rollback_costs={item.atom_id: item.rollback_cost for item in case.candidates},
            human_approval_costs={
                item.atom_id: item.human_approval_cost for item in case.candidates
            },
            fixed_costs={item.atom_id: item.fixed_cost for item in case.candidates},
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "atom_costs": {key: str(value) for key, value in self.atom_costs.items()},
            "action_costs": {key: str(value) for key, value in self.action_costs.items()},
            "rollback_costs": {
                key: str(value) for key, value in self.rollback_costs.items()
            },
            "human_approval_costs": {
                key: str(value) for key, value in self.human_approval_costs.items()
            },
            "fixed_costs": {key: str(value) for key, value in self.fixed_costs.items()},
        }

    def global_scale(self, multiplier: object) -> CostRegistry:
        factor = _decimal(multiplier)
        if factor <= 0:
            raise ValueError("global cost scale must be positive")
        return CostRegistry(
            atom_costs={key: value * factor for key, value in self.atom_costs.items()},
            action_costs={key: value * factor for key, value in self.action_costs.items()},
            rollback_costs={
                key: value * factor for key, value in self.rollback_costs.items()
            },
            human_approval_costs={
                key: value * factor for key, value in self.human_approval_costs.items()
            },
            fixed_costs={key: value * factor for key, value in self.fixed_costs.items()},
            schema_version=self.schema_version,
        )

    def non_uniform_scale(
        self,
        case: Case,
        *,
        node_multiplier: object,
        edge_multiplier: object,
        human_multiplier: object,
    ) -> CostRegistry:
        node_factor = _decimal(node_multiplier)
        edge_factor = _decimal(edge_multiplier)
        human_factor = _decimal(human_multiplier)
        if min(node_factor, edge_factor, human_factor) <= 0:
            raise ValueError("non-uniform cost multipliers must be positive")
        kinds = {item.atom_id: item.subject_kind for item in case.candidates}

        def factor(atom_id: str) -> Decimal:
            return edge_factor if kinds[atom_id] == "edge" else node_factor

        return CostRegistry(
            atom_costs={
                key: value * factor(key) for key, value in self.atom_costs.items()
            },
            action_costs={
                key: value * factor(key) for key, value in self.action_costs.items()
            },
            rollback_costs={
                key: value * factor(key) for key, value in self.rollback_costs.items()
            },
            human_approval_costs={
                key: value * human_factor
                for key, value in self.human_approval_costs.items()
            },
            fixed_costs=dict(self.fixed_costs),
            schema_version=self.schema_version,
        )


def apply_registry(case: Case, registry: CostRegistry) -> Case:
    expected = {item.atom_id for item in case.candidates}
    for values in (
        registry.atom_costs,
        registry.action_costs,
        registry.rollback_costs,
        registry.human_approval_costs,
        registry.fixed_costs,
    ):
        if set(values) != expected:
            raise ValueError("cost registry does not cover every candidate exactly once")
    return replace(
        case,
        candidates=tuple(
            replace(
                item,
                cost=float(registry.atom_costs[item.atom_id]),
                action_cost=float(registry.action_costs[item.atom_id]),
                rollback_cost=float(registry.rollback_costs[item.atom_id]),
                human_approval_cost=float(
                    registry.human_approval_costs[item.atom_id]
                ),
                fixed_cost=float(registry.fixed_costs[item.atom_id]),
            )
            for item in case.candidates
        ),
    )


def registry_diff(
    base: CostRegistry,
    transformed: CostRegistry,
    *,
    transformation_function: str,
    multiplier: str | None = None,
    perturbation_id: str | None = None,
) -> dict[str, object]:
    affected = {}
    for group in (
        "atom_costs",
        "action_costs",
        "rollback_costs",
        "human_approval_costs",
        "fixed_costs",
    ):
        before = getattr(base, group)
        after = getattr(transformed, group)
        affected[group] = [
            key for key in before if before[key] != after[key]
        ]
    return {
        "base_registry_sha256": base.sha256,
        "transformed_registry_sha256": transformed.sha256,
        "multiplier": multiplier,
        "perturbation_id": perturbation_id,
        "transformation_function": transformation_function,
        "components_transformed": [
            "atom_costs",
            "action_costs",
            "rollback_costs",
            "human_approval_costs",
            "fixed_costs",
        ],
        "affected": affected,
    }


def cost_cache_key(
    *,
    case_sha256: str,
    method_sha256: str,
    cost_registry_sha256: str,
    protocol_sha256: str,
    solver_config_sha256: str,
) -> str:
    payload = {
        "case_sha256": case_sha256,
        "method_sha256": method_sha256,
        "cost_registry_sha256": cost_registry_sha256,
        "protocol_sha256": protocol_sha256,
        "solver_config_sha256": solver_config_sha256,
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
