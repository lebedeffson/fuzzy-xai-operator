"""Build an action-conditioned certificate from observable route artifacts."""

from __future__ import annotations

from fuzzyxai.practical_controller import DeploymentContext, ExplanationArtifact, PredictionArtifact, RouteArtifacts

from .certificate import ActionConditionedAuditCertificate
from .contracts import ContractCheck, ContractOutcome, ContractRequirement


def build_action_certificate(
    prediction: PredictionArtifact,
    explanation: ExplanationArtifact,
    route: RouteArtifacts,
    context: DeploymentContext,
    *,
    action: str = "accept",
) -> ActionConditionedAuditCertificate:
    requirements: list[tuple[ContractRequirement, str | None]] = [
        (_require("model_version", context.expected_model_version, "prediction/model", 1.0), prediction.model_version),
        (_require("explainer_model_pair", context.expected_model_version, "explanation/model", 1.0), explanation.model_version),
        (_require("preprocessing_version", context.expected_preprocessing_version, "route/preprocessing", 1.0), route.preprocessing_version),
        (_require("explainer_version", context.expected_explainer_version, "explanation/explainer", 0.8), explanation.explainer_version),
        (_require("calibration_artifact", context.expected_calibration_version, "route/calibration", 0.8), route.calibration_version),
        (_require("reference_population", context.expected_reference_population, "route/reference", 0.8), route.reference_population),
        (_require("feature_schema", context.expected_schema_version, "route/schema", 1.0), route.schema_version),
        (_require("explain_plan", context.expected_explain_plan_version, "explanation/plan", 0.8), explanation.explain_plan_version),
        (_require("dictionary_version", context.expected_dictionary_version, "explanation/dictionary", 0.6), explanation.dictionary_version),
        (_require("route_artifact_integrity", context.expected_artifact_sha256, "route/artifact", 1.0), route.artifact_sha256),
        (_require("canonical_integrity", context.expected_artifact_sha256, "explanation/canonical", 1.0), explanation.canonical_sha256),
    ]
    observed = set(route.observed_provenance_channels)
    for channel in context.mandatory_provenance_channels:
        requirements.append(
            (
                _require(f"provenance:{channel}", "present", f"route/provenance/{channel}", 0.9),
                "present" if channel in observed else "missing",
            )
        )
    checks = [_check(requirement, actual) for requirement, actual in requirements]
    checks.extend(_boolean_checks(explanation, route, context))
    blocking_ok = all(check.satisfied or not check.requirement.blocking for check in checks)
    return ActionConditionedAuditCertificate(
        action=action,
        checks=tuple(checks),
        required_evidence=tuple(context.mandatory_provenance_channels),
        uncertainty_constraints=("predictive_risk_calibrated", "risk_interval_available"),
        representation_constraints=(f"reduction_loss<={context.maximum_reduction_loss}",),
        action_preconditions=("hard_contracts_satisfied", "review_budget_respected"),
        source_paths=tuple(check.requirement.source_path for check in checks),
        certificate_exists=blocking_ok,
    )


def _require(contract_id: str, expected: str, path: str, severity: float) -> ContractRequirement:
    return ContractRequirement(contract_id=contract_id, expected=expected, source_path=path, severity=severity)


def _check(requirement: ContractRequirement, actual: str | None) -> ContractCheck:
    satisfied = actual == requirement.expected
    return ContractCheck(
        requirement=requirement,
        actual=actual,
        outcome=ContractOutcome.SATISFIED if satisfied else ContractOutcome.UNSATISFIED,
        reason_code="OK" if satisfied else f"CONTRACT_FAILED:{requirement.contract_id}",
    )


def _boolean_checks(
    explanation: ExplanationArtifact,
    route: RouteArtifacts,
    context: DeploymentContext,
) -> list[ContractCheck]:
    values = [
        ("reduction_loss", explanation.representation_loss <= context.maximum_reduction_loss, "explanation/reduction", 0.7),
        ("forbidden_rule_conflict", not route.forbidden_rule_conflict, "route/rules", 1.0),
        ("critical_data_quality", not route.critical_data_quality_fault, "route/data_quality", 1.0),
    ]
    if route.route_fault_type:
        values.extend((f"route_fault:{item}", False, f"route/fault/{item}", 0.9) for item in route.route_fault_type.split("|"))
    else:
        values.append(("registered_route_fault", True, "route/fault", 0.9))
    if route.natural_failure:
        values.extend((f"natural_failure:{item}", False, f"route/runtime/{item}", 0.9) for item in route.natural_failure.split("|"))
    else:
        values.append(("natural_pipeline_failure", True, "route/runtime", 0.9))
    return [
        _check(_require(contract_id, "valid", path, severity), "valid" if valid else "invalid")
        for contract_id, valid, path, severity in values
    ]
