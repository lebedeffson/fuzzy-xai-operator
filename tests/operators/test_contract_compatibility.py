import pytest
from fuzzyxai.operators import (
    ComposedDiagnosticOperator,
    FunctionalDiagnosticOperator,
    OperatorContract,
    OperatorExecutionError,
)


def test_incompatible_input_output_contracts_fail_structurally() -> None:
    first = FunctionalDiagnosticOperator(
        "first",
        OperatorContract("in"),
        OperatorContract("out", produced_fields=frozenset({"artifact"})),
        lambda state: state,
    )
    second = FunctionalDiagnosticOperator(
        "second",
        OperatorContract("in-2", required_fields=frozenset({"model_version"})),
        OperatorContract("out-2"),
        lambda state: state,
    )

    with pytest.raises(OperatorExecutionError) as error:
        ComposedDiagnosticOperator(first, second)

    assert error.value.code == "OPERATOR_CONTRACT_MISMATCH"
