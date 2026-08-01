from __future__ import annotations

import re
from dataclasses import dataclass
from time import perf_counter_ns

from .formal_operations import FormalOperation, OperationEvent
from .repository_importer import IncidentRoute

CONTRACT_RULES: dict[str, tuple[str, ...]] = {
    "DEPENDENCY_VERSION": (
        "dependency",
        "dependencies",
        "version",
        "requirements",
        "compatibility",
        "deprecated",
    ),
    "MODEL_EXPLAINER_VERSION": ("explainer", "shap", "lime", "model version"),
    "PREPROCESSING_SCHEMA": (
        "preprocess",
        "transform",
        "feature names",
        "column",
        "dtype",
        "shape",
    ),
    "DATA_CONTRACT": (
        "dataframe",
        "dataset",
        "array",
        "index",
        "dimension",
        "coordinate",
        "missing value",
    ),
    "ARTIFACT_PROVENANCE": (
        "artifact",
        "metadata",
        "provenance",
        "cache",
        "path",
        "file",
    ),
    "SERIALIZATION": (
        "serialize",
        "serialization",
        "pickle",
        "json",
        "yaml",
        "csv",
        "save",
    ),
    "CHECKSUM_INTEGRITY": ("checksum", "digest", "hash mismatch", "corrupt"),
    "PIPELINE_CONFIGURATION": (
        "configuration",
        "config",
        "setting",
        "option",
        "parameter",
    ),
    "MODEL_LOADING": (
        "load model",
        "model loading",
        "checkpoint",
        "state_dict",
        "estimator",
    ),
    "EXPLAINER_CONFIGURATION": (
        "explainer configuration",
        "background data",
        "attribution",
        "explanation",
    ),
}

COMPONENT_RULES: dict[str, tuple[str, ...]] = {
    "dependencies": ("dependency", "requirements", "version", "package", "install"),
    "data_schema": ("schema", "column", "dtype", "shape", "dimension", "index"),
    "preprocessing": ("preprocess", "transform", "encoding", "normaliz", "feature"),
    "model": ("model", "estimator", "classifier", "regressor", "predict"),
    "explainer": ("explainer", "shap", "lime", "attribution", "explanation"),
    "artifact": ("artifact", "serialize", "pickle", "json", "yaml", "save", "cache"),
    "configuration": ("config", "setting", "option", "parameter"),
    "runtime": ("runtime", "exception", "error", "traceback"),
}


@dataclass(frozen=True)
class IncidentPrediction:
    method: str
    source_component: str
    contract_family: str
    predicted_files: tuple[str, ...]
    repair_operation: str | None
    abstained: bool
    operations: tuple[OperationEvent, ...]


def _event(
    operation: FormalOperation,
    started: int,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    dependency_cost: float = 0.0,
) -> OperationEvent:
    return OperationEvent(
        operation_id=operation,
        precondition="observable_evidence_available",
        input_evidence=inputs,
        output_evidence=outputs,
        measured_runtime_ms=(perf_counter_ns() - started) / 1_000_000,
        dependency_cost=dependency_cost,
    )


def _scores(text: str, rules: dict[str, tuple[str, ...]]) -> dict[str, int]:
    lowered = text.lower()
    return {
        key: sum(lowered.count(token) for token in tokens)
        for key, tokens in rules.items()
    }


def _best(scores: dict[str, int], default: str) -> str:
    best_value = max(scores.values(), default=0)
    if best_value <= 0:
        return default
    return min(key for key, value in scores.items() if value == best_value)


def _files(text: str) -> tuple[str, ...]:
    matches = re.findall(
        r"(?:^|[\s`'\"(])([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+\.(?:py|json|ya?ml|csv|toml|cfg|ini))",
        text,
    )
    return tuple(dict.fromkeys(matches))[:3]


def _repair(contract: str) -> str | None:
    return {
        "DEPENDENCY_VERSION": "dependency_pin_update",
        "MODEL_EXPLAINER_VERSION": "model_explainer_version_alignment",
        "PREPROCESSING_SCHEMA": "schema_synchronization",
        "DATA_CONTRACT": "schema_synchronization",
        "ARTIFACT_PROVENANCE": "artifact_regeneration",
        "SERIALIZATION": "serialization_format_correction",
        "CHECKSUM_INTEGRITY": "checksum_restoration",
        "PIPELINE_CONFIGURATION": "configuration_correction",
        "MODEL_LOADING": "pipeline_stage_rerun",
        "EXPLAINER_CONFIGURATION": "configuration_correction",
    }.get(contract)


def audit_traceback(route: IncidentRoute) -> IncidentPrediction:
    started = perf_counter_ns()
    files = _files(route.observable_text)
    component = "runtime"
    contract = "PIPELINE_CONFIGURATION"
    operations = (
        _event(FormalOperation.OPEN_LOG, started, ("failing_test",), ("traceback",)),
        _event(FormalOperation.SELECT_COMPONENT, started, ("traceback",), (component,)),
    )
    return IncidentPrediction("B_TRACE", component, contract, files[:1], _repair(contract), False, operations)


def audit_rule(route: IncidentRoute) -> IncidentPrediction:
    started = perf_counter_ns()
    contract_scores = _scores(route.observable_text, CONTRACT_RULES)
    contract = next((key for key, value in contract_scores.items() if value > 0), "")
    if not contract:
        return IncidentPrediction("B_RULE", "", "", (), None, True, ())
    component = _best(_scores(route.observable_text, COMPONENT_RULES), "configuration")
    operations = (
        _event(FormalOperation.OPEN_MANIFEST, started, ("issue",), ("rule_input",)),
        _event(FormalOperation.FORM_HYPOTHESIS, started, ("rule_input",), (contract,)),
        _event(FormalOperation.SELECT_COMPONENT, started, (contract,), (component,)),
    )
    return IncidentPrediction("B_RULE", component, contract, _files(route.observable_text), _repair(contract), False, operations)


def audit_greedy(route: IncidentRoute) -> IncidentPrediction:
    started = perf_counter_ns()
    component_scores = _scores(route.observable_text, COMPONENT_RULES)
    component = _best(component_scores, "runtime")
    contract = _best(_scores(route.observable_text, CONTRACT_RULES), "PIPELINE_CONFIGURATION")
    operations = (
        _event(FormalOperation.OPEN_LOG, started, ("issue", "failing_test"), ("tokens",)),
        _event(FormalOperation.TRACE_DEPENDENCY, started, ("tokens",), (component,), 1.0),
        _event(FormalOperation.FORM_HYPOTHESIS, started, (component,), (contract,)),
        _event(FormalOperation.SELECT_COMPONENT, started, (contract,), (component,)),
    )
    return IncidentPrediction("B_GREEDY", component, contract, _files(route.observable_text), _repair(contract), False, operations)


def audit_route(route: IncidentRoute) -> IncidentPrediction:
    started = perf_counter_ns()
    text = route.observable_text
    contract_scores = _scores(text, CONTRACT_RULES)
    component_scores = _scores(text, COMPONENT_RULES)
    lowered = text.lower()
    if "load" in lowered and "model" in lowered:
        contract_scores["MODEL_LOADING"] += 3
        component_scores["model"] += 2
    if any(token in lowered for token in ("dtype", "column", "shape", "dimension")):
        contract_scores["PREPROCESSING_SCHEMA"] += 2
        component_scores["data_schema"] += 2
    if any(token in lowered for token in ("pickle", "serialize", "save", "json")):
        contract_scores["SERIALIZATION"] += 2
        component_scores["artifact"] += 2
    if any(token in lowered for token in ("version", "deprecated", "requirements")):
        contract_scores["DEPENDENCY_VERSION"] += 2
        component_scores["dependencies"] += 2
    contract = _best(contract_scores, "PIPELINE_CONFIGURATION")
    component = _best(component_scores, "configuration")
    operations = (
        _event(FormalOperation.OPEN_LOG, started, ("failing_test",), ("runtime_evidence",)),
        _event(FormalOperation.OPEN_MANIFEST, started, ("repository",), ("manifest_evidence",)),
        _event(FormalOperation.READ_VERSION, started, ("manifest_evidence",), ("version",)),
        _event(FormalOperation.READ_SCHEMA, started, ("issue",), ("schema_terms",)),
        _event(FormalOperation.TRACE_DEPENDENCY, started, ("route_graph",), (component,), 2.0),
        _event(FormalOperation.COMPARE_SCHEMA, started, ("schema_terms",), (contract,)),
        _event(FormalOperation.FORM_HYPOTHESIS, started, (contract,), ("typed_hypothesis",)),
        _event(FormalOperation.SELECT_COMPONENT, started, ("typed_hypothesis",), (component,)),
    )
    return IncidentPrediction("O_ROUTE", component, contract, _files(text), _repair(contract), False, operations)
