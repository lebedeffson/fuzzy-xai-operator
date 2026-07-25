from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True)
class RepairAction:
    action_id: str
    target_component: str
    covers: frozenset[str]
    action_kind: str
    dependency_fanout: int
    runtime_units: int
    rollback_risk: float
    dependencies: tuple[str, ...] = ()
    additional_touched_components: tuple[str, ...] = ()
    modified_artifacts: int = 1
    direct_repair: bool = True
    creates_critical_violation: bool = False

    def __post_init__(self) -> None:
        if not self.action_id or not self.target_component:
            raise ValueError("repair action identifiers cannot be empty")
        if not self.covers:
            raise ValueError("repair action must cover at least one obligation")
        if self.dependency_fanout < 0 or self.runtime_units < 1:
            raise ValueError("fan-out and runtime units must be non-negative")
        if not 0.0 <= self.rollback_risk <= 1.0:
            raise ValueError("rollback risk must be in [0, 1]")


@dataclass(frozen=True)
class StrategyPlan:
    strategy: str
    action_ids: tuple[str, ...]
    predicted_cost: float
    covered_obligations: tuple[str, ...]
    feasible: bool
    equivalent_optimal_plans: tuple[tuple[str, ...], ...] = ()


ActionCost = Callable[[RepairAction], float]


def _ordered(
    actions: Iterable[RepairAction],
    action_ids: Iterable[str],
) -> tuple[RepairAction, ...]:
    by_id = {action.action_id: action for action in actions}
    requested = set(action_ids)
    missing = requested - set(by_id)
    if missing:
        raise KeyError(f"unknown repair actions: {sorted(missing)}")
    selected = [action for action in actions if action.action_id in requested]
    emitted = {action.action_id for action in selected}
    while True:
        dependencies = {
            dependency
            for action in selected
            for dependency in action.dependencies
            if dependency not in emitted
        }
        if not dependencies:
            break
        additions = [action for action in actions if action.action_id in dependencies]
        if len(additions) != len(dependencies):
            raise ValueError("repair plan references unavailable dependencies")
        selected = additions + selected
        emitted.update(action.action_id for action in additions)
    return tuple(dict.fromkeys(selected))


def _coverage(actions: Iterable[RepairAction]) -> frozenset[str]:
    return frozenset(obligation for action in actions for obligation in action.covers)


def _is_feasible(
    selected: tuple[RepairAction, ...],
    obligations: frozenset[str],
) -> bool:
    ids = {action.action_id for action in selected}
    return (
        obligations.issubset(_coverage(selected))
        and all(set(action.dependencies).issubset(ids) for action in selected)
        and not any(action.creates_critical_violation for action in selected)
    )


def enumerate_valid_repair_sets(
    actions: tuple[RepairAction, ...],
    obligations: frozenset[str],
) -> tuple[tuple[RepairAction, ...], ...]:
    valid: list[tuple[RepairAction, ...]] = []
    for size in range(1, len(actions) + 1):
        for subset in combinations(actions, size):
            ordered = _ordered(actions, (action.action_id for action in subset))
            if _is_feasible(ordered, obligations) and ordered not in valid:
                valid.append(ordered)
    return tuple(valid)


def _plan(
    strategy: str,
    selected: tuple[RepairAction, ...],
    obligations: frozenset[str],
    cost: ActionCost,
    *,
    equivalent: tuple[tuple[str, ...], ...] = (),
) -> StrategyPlan:
    covered = _coverage(selected)
    return StrategyPlan(
        strategy=strategy,
        action_ids=tuple(action.action_id for action in selected),
        predicted_cost=sum(cost(action) for action in selected),
        covered_obligations=tuple(sorted(covered)),
        feasible=_is_feasible(selected, obligations),
        equivalent_optimal_plans=equivalent,
    )


def select_repair_all(
    actions: tuple[RepairAction, ...],
    obligations: frozenset[str],
    cost: ActionCost,
) -> StrategyPlan:
    direct = tuple(
        action
        for action in actions
        if action.direct_repair and action.covers.intersection(obligations)
    )
    selected = _ordered(actions, (action.action_id for action in direct))
    return _plan("B_ALL", selected, obligations, cost)


def select_first_valid(
    actions: tuple[RepairAction, ...],
    obligations: frozenset[str],
    cost: ActionCost,
) -> StrategyPlan:
    prefix: list[RepairAction] = []
    for action in actions:
        if action.creates_critical_violation:
            continue
        prefix.append(action)
        selected = _ordered(actions, (item.action_id for item in prefix))
        if _is_feasible(selected, obligations):
            return _plan("B_FIRST", selected, obligations, cost)
    valid = enumerate_valid_repair_sets(actions, obligations)
    selected = valid[0] if valid else ()
    return _plan("B_FIRST", selected, obligations, cost)


def select_local_greedy(
    actions: tuple[RepairAction, ...],
    obligations: frozenset[str],
    cost: ActionCost,
) -> StrategyPlan:
    remaining = set(obligations)
    selected: list[RepairAction] = []
    while remaining:
        useful = [
            action
            for action in actions
            if action.action_id not in {item.action_id for item in selected}
            and remaining.intersection(action.covers)
        ]
        if not useful:
            break
        chosen = min(
            useful,
            key=lambda action: (
                cost(action) / len(remaining.intersection(action.covers)),
                cost(action),
                action.action_id,
            ),
        )
        selected.extend(
            action
            for action in _ordered(actions, (chosen.action_id,))
            if action.action_id not in {item.action_id for item in selected}
        )
        remaining.difference_update(_coverage(selected))
    ordered = _ordered(actions, (action.action_id for action in selected))
    return _plan("B_GREEDY", ordered, obligations, cost)


def select_global_minimum_cut(
    actions: tuple[RepairAction, ...],
    obligations: frozenset[str],
    cost: ActionCost,
    *,
    tolerance: float = 1e-9,
) -> StrategyPlan:
    valid = enumerate_valid_repair_sets(actions, obligations)
    if not valid:
        return _plan("O_GLOBAL", (), obligations, cost)
    costs = tuple(sum(cost(action) for action in plan) for plan in valid)
    optimum = min(costs)
    equivalent = tuple(
        tuple(action.action_id for action in plan)
        for plan, value in zip(valid, costs)
        if abs(value - optimum) <= tolerance
    )
    selected_ids = min(equivalent)
    selected = _ordered(actions, selected_ids)
    return _plan(
        "O_GLOBAL",
        selected,
        obligations,
        cost,
        equivalent=tuple(sorted(equivalent)),
    )
