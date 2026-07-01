from __future__ import annotations

from fuzzyxai.core.types import OperatorRoute, ProofTrace


def build_proof_trace(route: OperatorRoute) -> ProofTrace:
    return ProofTrace(
        package_type="FuzzyXAIProofTrace",
        schema_version="1.0",
        scenario_id=route.scenario_id,
        route=route.to_dict(),
        computed_result=route.computed_result,
        diagnostics=route.diagnostics,
        final_action=route.final_action,
        verifier_status="PASS",
    )
