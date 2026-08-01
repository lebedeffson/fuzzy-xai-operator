import pytest
from fuzzyxai.operators import (
    FunctionalDiagnosticOperator,
    OperatorContract,
    OperatorExecutionError,
    RouteState,
    compose_operators,
)


def test_missing_provenance_fails_before_execution() -> None:
    operator = FunctionalDiagnosticOperator(
        "ValidateArtifactProvenance",
        OperatorContract(
            "provenance-input",
            required_fields=frozenset({"artifact", "provenance"}),
            produced_fields=frozenset({"validated"}),
        ),
        OperatorContract("validated", produced_fields=frozenset({"validated"})),
        lambda state: state.updated("ValidateArtifactProvenance", validated=True),
    )

    with pytest.raises(OperatorExecutionError) as error:
        compose_operators(operator).apply(RouteState(values={"artifact": "a"}))

    assert error.value.code == "OPERATOR_PRECONDITION_FAILED"
