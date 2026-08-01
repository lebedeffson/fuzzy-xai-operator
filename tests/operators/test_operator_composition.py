from fuzzyxai.operators import (
    FunctionalDiagnosticOperator,
    OperatorContract,
    RouteState,
    compose_operators,
)


def test_compatible_composition_executes_in_order() -> None:
    collect = FunctionalDiagnosticOperator(
        "CollectExplanationArtifact",
        OperatorContract("raw", frozenset({"raw"}), frozenset({"artifact"})),
        OperatorContract(
            "artifact",
            frozenset({"raw"}),
            frozenset({"artifact", "provenance"}),
        ),
        lambda state: state.updated(
            "CollectExplanationArtifact",
            artifact="artifact-1",
            provenance="registry://artifact-1",
        ),
    )
    validate = FunctionalDiagnosticOperator(
        "ValidateArtifactProvenance",
        OperatorContract(
            "artifact-input",
            frozenset({"artifact", "provenance"}),
            frozenset(),
        ),
        OperatorContract(
            "validated",
            frozenset(),
            frozenset({"artifact", "provenance", "provenance_valid"}),
        ),
        lambda state: state.updated(
            "ValidateArtifactProvenance",
            provenance_valid=True,
        ),
    )

    result = compose_operators(collect, validate).apply(
        RouteState(values={"raw": b"explanation"})
    )

    assert result.values["provenance_valid"] is True
    assert result.operator_trace == (
        "CollectExplanationArtifact",
        "ValidateArtifactProvenance",
    )
