import pytest
from fuzzyxai.operators import (
    FunctionalDiagnosticOperator,
    OperatorContract,
    OperatorExecutionError,
    RouteState,
    compose_operators,
)


def test_operator_exception_is_structured() -> None:
    def fail(state: RouteState) -> RouteState:
        del state
        raise RuntimeError("provider unavailable")

    operator = FunctionalDiagnosticOperator(
        "ExecuteRepairPlan",
        OperatorContract("plan", required_fields=frozenset({"plan"})),
        OperatorContract("result", produced_fields=frozenset({"result"})),
        fail,
    )

    with pytest.raises(OperatorExecutionError) as error:
        compose_operators(operator).apply(RouteState(values={"plan": "p"}))

    assert error.value.code == "OPERATOR_EXECUTION_FAILED"


def test_new_critical_violation_fails_postcondition() -> None:
    operator = FunctionalDiagnosticOperator(
        "RecertifyRoute",
        OperatorContract("route", required_fields=frozenset({"route"})),
        OperatorContract("recertified", produced_fields=frozenset({"recertified"})),
        lambda state: RouteState(
            values={**state.values, "recertified": False},
            critical_violations=("new-critical",),
            operator_trace=(*state.operator_trace, "RecertifyRoute"),
        ),
    )

    with pytest.raises(OperatorExecutionError) as error:
        compose_operators(operator).apply(RouteState(values={"route": "r"}))

    assert error.value.code == "OPERATOR_POSTCONDITION_FAILED"
