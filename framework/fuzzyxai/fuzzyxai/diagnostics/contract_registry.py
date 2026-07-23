from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from .contracts import Contract, RouteEdge, RouteGraph, RouteNode


@dataclass(frozen=True)
class ContractCheck:
    passed: bool
    insufficient_evidence: bool
    actual: object | None
    message: str


ContractEvaluator = Callable[[Contract, RouteNode | RouteEdge | None, RouteGraph], ContractCheck]


def _attributes(subject: RouteNode | RouteEdge | None) -> dict[str, object]:
    if isinstance(subject, RouteNode):
        return subject.observed_attributes
    if isinstance(subject, RouteEdge):
        return subject.observed_contract
    return {}


def _node_present(contract: Contract, subject: RouteNode | RouteEdge | None, _: RouteGraph) -> ContractCheck:
    return ContractCheck(subject is not None, False, subject is not None, "обязательный узел присутствует")


def _edge_present(contract: Contract, subject: RouteNode | RouteEdge | None, _: RouteGraph) -> ContractCheck:
    return ContractCheck(subject is not None, False, subject is not None, "обязательное ребро присутствует")


def _required_attribute(contract: Contract, subject: RouteNode | RouteEdge | None, _: RouteGraph) -> ContractCheck:
    attributes = _attributes(subject)
    actual = attributes.get(str(contract.field))
    missing = actual in (None, "", (), [], {})
    return ContractCheck(not missing, missing, actual, "обязательное значение доступно")


def _equals(contract: Contract, subject: RouteNode | RouteEdge | None, _: RouteGraph) -> ContractCheck:
    attributes = _attributes(subject)
    field = str(contract.field)
    if field not in attributes or attributes[field] in (None, ""):
        return ContractCheck(False, True, None, "наблюдаемое значение отсутствует")
    actual = attributes[field]
    return ContractCheck(actual == contract.expected, False, actual, "наблюдаемое значение соответствует зарегистрированному")


def _maximum(contract: Contract, subject: RouteNode | RouteEdge | None, _: RouteGraph) -> ContractCheck:
    attributes = _attributes(subject)
    field = str(contract.field)
    if field not in attributes or attributes[field] is None:
        return ContractCheck(False, True, None, "наблюдаемое значение отсутствует")
    actual = float(attributes[field])
    expected = float(contract.expected)
    return ContractCheck(actual <= expected, False, actual, "значение не превышает зарегистрированный предел")


def _minimum(contract: Contract, subject: RouteNode | RouteEdge | None, _: RouteGraph) -> ContractCheck:
    attributes = _attributes(subject)
    field = str(contract.field)
    if field not in attributes or attributes[field] is None:
        return ContractCheck(False, True, None, "наблюдаемое значение отсутствует")
    actual = float(attributes[field])
    expected = float(contract.expected)
    return ContractCheck(actual >= expected, False, actual, "значение достигает зарегистрированного минимума")


def _allowed(contract: Contract, subject: RouteNode | RouteEdge | None, _: RouteGraph) -> ContractCheck:
    attributes = _attributes(subject)
    field = str(contract.field)
    if field not in attributes:
        return ContractCheck(False, True, None, "наблюдаемое значение отсутствует")
    actual = attributes[field]
    allowed = tuple(contract.parameters.get("allowed", ()))
    return ContractCheck(actual in allowed, False, actual, "значение входит в зарегистрированное множество")


@dataclass
class ContractRegistry:
    _evaluators: dict[str, ContractEvaluator] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "ContractRegistry":
        return cls(
            {
                "node_present": _node_present,
                "edge_present": _edge_present,
                "required_attribute": _required_attribute,
                "equals": _equals,
                "max_value": _maximum,
                "min_value": _minimum,
                "allowed": _allowed,
                "checksum": _equals,
                "compatible": _equals,
            }
        )

    def register(self, kind: str, evaluator: ContractEvaluator) -> None:
        if not kind:
            raise ValueError("contract kind cannot be empty")
        self._evaluators[kind] = evaluator

    def evaluate(self, contract: Contract, graph: RouteGraph) -> ContractCheck:
        subject: RouteNode | RouteEdge | None = graph.node(contract.subject_id) or graph.edge(contract.subject_id)
        evaluator = self._evaluators.get(contract.kind)
        if evaluator is None:
            return ContractCheck(False, True, None, f"неизвестный вид контракта: {contract.kind}")
        try:
            return evaluator(contract, subject, graph)
        except (TypeError, ValueError) as exc:
            return ContractCheck(False, True, None, f"контракт не может быть проверен: {exc}")
