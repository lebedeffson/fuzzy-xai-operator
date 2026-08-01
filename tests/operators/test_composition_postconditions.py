import pytest
from fuzzyxai.operators import (
    FunctionalDiagnosticOperator,
    OperatorContract,
    OperatorExecutionError,
    RouteState,
    compose_operators,
)


def test_postcondition_failure_is_fail_closed() -> None:
    operator = FunctionalDiagnosticOperator(
        "BuildDiagnosticGraph",
        OperatorContract(
            "artifact",
            required_fields=frozenset({"artifact"}),
            produced_fields=frozenset({"graph"}),
        ),
        OperatorContract("graph", produced_fields=frozenset({"graph"})),
        lambda state: state.updated("BuildDiagnosticGraph"),
    )

    with pytest.raises(OperatorExecutionError) as error:
        compose_operators(operator).apply(RouteState(values={"artifact": "a"}))

    assert error.value.code == "OPERATOR_POSTCONDITION_FAILED"
