"""Developmental matched-information route audit using three fitted sklearn datasets.

Runs trained models OUTSIDE the FuzzyXAI verifier.  Faults are deliberately
specified; comparisons are not independent natural-incident validation.
The equal-evidence manual comparator is included to avoid unfair claims.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import numpy as np
import sklearn
from fuzzyxai.core.certified_route import (
    RouteStatus,
    SignedPrediction,
    certify_branching_route,
    certify_route,
    sign_prediction,
    verify_prediction,
)
from fuzzyxai.core.composition import compose
from fuzzyxai.core.diagnostics import DiagnosticState
from fuzzyxai.core.explanation_object import ExplanationObject, Rule, Trace
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Fixed development protocol, not a hidden confirmatory study.
SEEDS = (17, 31, 53)
TEST_KEY = b"development-only-shared-key-not-for-deployment"
TOL = 1e-8
BETA = {"repr": 0.2, "rules": 0.2, "activations": 0.2,
        "uncertainty": 0.15, "trace": 0.15, "reduction": 0.1}
CASES = (
    "clean", "valid_complement", "coherent_event_spoof",
    "coherent_value_spoof", "coherent_source_spoof", "coherent_model_spoof",
    "unregistered_complement", "rule_collision", "stale_trace",
    "forged_signed_reference", "missing_signed_reference",
    "missing_semantic_contract", "ground_truth_label_permutation",
)


class ScalarRep:
    def __init__(self, v: float):
        self.value = float(v)

    def distance(self, other: object) -> float:
        return abs(self.value - other.value)


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")


def _digest(obj) -> str:
    return hashlib.sha256(_canonical(obj)).hexdigest()


def _model_hash(model) -> str:
    sc, lr = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
    return _digest({"coef": lr.coef_.tolist(), "intercept": lr.intercept_.tolist(),
                    "classes": lr.classes_.tolist(), "scaler_mean": sc.mean_.tolist(),
                    "scaler_scale": sc.scale_.tolist()})


def _input_hash(x) -> str:
    # Byte ordering and dimensionality are explicit in the canonicalization.
    return _digest({"schema": "numeric-tabular-input-v1", "feature_names":
                    tuple(f"feature_{i}" for i in range(len(x))),
                    "values": tuple(float(v) for v in x)})


def _datasets():
    x, y = load_breast_cancer(return_X_y=True)
    yield "breast_cancer", x, (y == 0).astype(int)
    x, y = load_wine(return_X_y=True)
    yield "wine_class_2", x, (y == 2).astype(int)
    x, y = load_digits(return_X_y=True)
    yield "digits_ge_5", x, (y >= 5).astype(int)


def _stage(anchor: SignedPrediction, n: int, *, p: float | None = None,
           event: str | None = None) -> ExplanationObject:
    p = anchor.score if p is None else float(p)
    e = anchor.event_id if event is None else event
    return ExplanationObject(
        terms={"positive", "negative"}, representation=ScalarRep(p),
        rules=[Rule(f"rule-{n}", {"decision": "review"}, "inspect")],
        activations={f"rule-{n}": 0.7}, uncertainty=0.11,
        trace=Trace(anchor.object_id, anchor.model_version, "2026-11-14",
                    source="independent-sklearn-source",
                    checksum=anchor.trace_checksum if n == 0 else f"stage-{n}-proof"),
        metadata={
            "risk": p,
            "semantic_contract": {
                "object_id": anchor.object_id, "event_id": e,
                "model_version": anchor.model_version, "unit": anchor.unit,
                "source_sha256": anchor.source_sha256,
                "model_sha256": anchor.model_sha256,
            },
        },
    )


def _mutate(anchor, case):
    reference = anchor
    steps = [_stage(anchor, n) for n in (0, 1, 2)]
    if case == "valid_complement":
        right = steps[-1]
        right.metadata["semantic_contract"]["event_id"] = "negative"
        right.metadata["risk"] = 1.0 - anchor.score
        right.representation.value = 1.0 - anchor.score
        right.metadata["semantic_transition"] = {
            "kind": "complement", "from_event": "positive", "to_event": "negative"}
    elif case == "coherent_event_spoof":
        for step in steps:
            step.metadata["semantic_contract"]["event_id"] = "negative"
    elif case == "coherent_value_spoof":
        new_p = min(1.0, anchor.score + 0.1) if anchor.score <= 0.8 else anchor.score - 0.1
        for step in steps:
            step.metadata["risk"] = new_p
    elif case == "coherent_source_spoof":
        for step in steps:
            step.metadata["semantic_contract"]["source_sha256"] = "f" * 64
    elif case == "coherent_model_spoof":
        for step in steps:
            step.metadata["semantic_contract"]["model_sha256"] = "f" * 64
    elif case == "unregistered_complement":
        steps[-1].metadata["semantic_contract"]["event_id"] = "negative"
        steps[-1].metadata["risk"] = 1.0 - anchor.score
    elif case == "rule_collision":
        steps[-1].rules = [Rule("rule-1", {"decision": "review"}, "approve")]
    elif case == "stale_trace":
        steps[0].trace = replace(steps[0].trace, checksum="different-trace")
    elif case == "forged_signed_reference":
        reference = replace(anchor, score=1.0 - anchor.score)
    elif case == "missing_signed_reference":
        reference = None
    elif case == "missing_semantic_contract":
        steps[-1].metadata.pop("semantic_contract")
    elif case in ("clean", "ground_truth_label_permutation"):
        pass
    else:
        raise ValueError("unregistered case")
    return reference, steps


def _baseline_pairwise(steps) -> str:
    # Existing compose without a source anchor. This is deliberately weaker.
    for left, right in pairwise(steps):
        output = compose(left, right, BETA)
        if isinstance(output, DiagnosticState):
            return RouteStatus.REJECTED.value
    return RouteStatus.CERTIFIED.value


def _manual_equal_evidence(reference, steps) -> str:
    # Independent manual decision code with the SAME source receipt and fields.
    # A sophisticated rule engine is expected to tie the research operator
    # here.  We never claim this comparator is inferior by construction.
    if reference is None or not reference.valid_shape():
        return RouteStatus.INSUFFICIENT.value
    if verify_prediction(reference, secret=TEST_KEY) is not RouteStatus.CERTIFIED:
        return RouteStatus.REJECTED.value
    old_ev, old_p = reference.event_id, reference.score
    ids = {}
    for i, item in enumerate(steps):
        contract = item.metadata.get("semantic_contract")
        if not isinstance(contract, dict) or any(
            k not in contract for k in (
                "object_id", "event_id", "model_version", "unit", "source_sha256", "model_sha256")
        ):
            return RouteStatus.INSUFFICIENT.value
        for key in ("object_id", "model_version", "unit", "source_sha256", "model_sha256"):
            if contract[key] != getattr(reference, key):
                return RouteStatus.REJECTED.value
        if not item.trace.source or not item.trace.checksum:
            return RouteStatus.INSUFFICIENT.value
        if i == 0 and item.trace.checksum != reference.trace_checksum:
            return RouteStatus.REJECTED.value
        p = item.metadata.get("risk")
        if not isinstance(p, (int, float)) or not np.isfinite(p) or not 0 <= p <= 1:
            return RouteStatus.INSUFFICIENT.value
        ev = contract["event_id"]
        t = item.metadata.get("semantic_transition")
        if ev == old_ev:
            if t is not None:
                return RouteStatus.REJECTED.value
            expected = old_p
        else:
            if i == 0 or not isinstance(t, dict) or t != {
                "kind": "complement", "from_event": old_ev, "to_event": ev}:
                return RouteStatus.REJECTED.value
            if reference.complement_event_id is None or {old_ev, ev} != {reference.event_id, reference.complement_event_id}:
                return RouteStatus.REJECTED.value
            expected = 1.0 - old_p
        if abs(p - expected) > TOL:
            return RouteStatus.REJECTED.value
        old_p, old_ev = p, ev
        for rule in item.rules:
            if rule.name in ids and ids[rule.name] != rule.signature():
                return RouteStatus.REJECTED.value
            ids[rule.name] = rule.signature()
    return RouteStatus.CERTIFIED.value


def run(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    graph_results = []
    fingerprints = []
    fitted = 0
    for dataset, x, y in _datasets():
        for seed in SEEDS:
            tr, te = train_test_split(
                np.arange(len(y)), random_state=seed, test_size=0.2, stratify=y,
            )
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1200, C=1.0, random_state=seed),
            )
            model.fit(x[tr], y[tr]); fitted += 1
            model_sha = _model_hash(model)
            fit_id = f"{dataset}-seed{seed}"
            fingerprints.append({"dataset": dataset, "seed": seed, "model_sha256": model_sha,
                                 "train_n": len(tr), "test_n": len(te)})
            pred = model.predict_proba(x[te])[:, 1]
            for example_order, (row_index, p) in enumerate(zip(te, pred, strict=True)):
                object_id = f"{dataset}:{int(row_index)}"
                source_sha = _input_hash(x[int(row_index)])
                anchor = sign_prediction(SignedPrediction(
                    object_id=object_id, event_id="positive", model_version=fit_id,
                    unit="probability", score=float(p),
                    source_sha256=source_sha, model_sha256=model_sha,
                    trace_checksum=hashlib.sha256((fit_id + object_id).encode()).hexdigest(),
                    complement_event_id="negative",
                ), secret=TEST_KEY)
                if example_order == 0:
                    forks = [("root", "a"), ("root", "b"),
                             ("a", "merge"), ("b", "merge")]
                    for graph_case in ("clean_fork_merge", "conflicting_branch",
                                       "missing_terminal", "path_limit"):
                        graph_nodes = {k: _stage(anchor, i) for i, k in enumerate(
                            ("root", "a", "b", "merge"))}
                        terms = ["merge"] if graph_case != "missing_terminal" else ["a"]
                        if graph_case == "conflicting_branch":
                            graph_nodes["b"].metadata["risk"] = (
                                anchor.score - .1 if anchor.score >= .1 else anchor.score + .1)
                        graph_result = certify_branching_route(
                            graph_nodes, forks, root_id="root", terminal_ids=terms,
                            signed_prediction=anchor, verification_secret=TEST_KEY,
                            beta=BETA, max_paths=1 if graph_case == "path_limit" else 16,
                        )
                        expected_graph = (
                            RouteStatus.CERTIFIED if graph_case == "clean_fork_merge"
                            else RouteStatus.REJECTED if graph_case == "conflicting_branch"
                            else RouteStatus.INSUFFICIENT
                        )
                        assert graph_result.status == expected_graph, (graph_case, graph_result)
                        graph_results.append({
                            "dataset": dataset, "seed": seed, "case": graph_case,
                            "expected": expected_graph.value, "observed": graph_result.status.value,
                            "finding": graph_result.findings[0].code if graph_result.findings else "OK",
                        })
                for case in CASES:
                    active_ref, route = _mutate(anchor, case)
                    observed = certify_route(route, signed_prediction=active_ref,
                                             verification_secret=TEST_KEY, beta=BETA,
                                             score_tolerance=TOL)
                    equal = _manual_equal_evidence(active_ref, route)
                    true_label = int(y[int(row_index)])
                    observed_label = (
                        1 - true_label if case == "ground_truth_label_permutation"
                        else true_label
                    )
                    results.append({
                        "observed_label": observed_label, "source_label": true_label,
                        "dataset": dataset, "seed": seed, "object_id": object_id,
                        "fault_family": case,
                        "expected": (RouteStatus.INSUFFICIENT.value
                                     if case in ("missing_signed_reference", "missing_semantic_contract")
                                     else RouteStatus.CERTIFIED.value
                                     if case in ("clean", "valid_complement", "ground_truth_label_permutation")
                                     else RouteStatus.REJECTED.value),
                        "with_signed_route": observed.status.value,
                        "pairwise_without_anchor": _baseline_pairwise(route),
                        "manual_equal_evidence": equal,
                        "reason": observed.findings[0].code if observed.findings else "OK",
                    })
    assert all(row["with_signed_route"] == row["expected"] for row in results)
    assert all((r["observed_label"] != r["source_label"]) == (r["fault_family"] == "ground_truth_label_permutation") for r in results)
    assert all(row["manual_equal_evidence"] == row["with_signed_route"] for row in results)
    assert fitted == 9
    with (output_dir / "graph_cases.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(graph_results[0])); writer.writeheader(); writer.writerows(graph_results)
    with (output_dir / "case_results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    by_case = []
    for case in CASES:
        sub = [row for row in results if row["fault_family"] == case]
        by_case.append({
            "case": case, "n": len(sub), "expected": sub[0]["expected"],
            "full": {s: sum(x["with_signed_route"] == s for x in sub) for s in ("certified", "rejected", "insufficient_evidence")},
            "pairwise_certified": sum(x["pairwise_without_anchor"] == "certified" for x in sub),
            "equal_evidence_matches": sum(x["manual_equal_evidence"] == x["with_signed_route"] for x in sub),
        })
    report = {
        "protocol_id": "signed-route-development-v1", "locked_seeds": list(SEEDS),
        "cases": list(CASES), "models": fitted, "n_records": len(results),
        "n_unique_test_predictions": len(results) // len(CASES),
        "graph_case_rows": len(graph_results),
        "graph_statuses": {case: {"observed": [r["observed"] for r in graph_results if r["case"] == case],
                                  "n": sum(r["case"] == case for r in graph_results)}
                            for case in ("clean_fork_merge", "conflicting_branch",
                                         "missing_terminal", "path_limit")},
        "coherent_attack_comparator": "pairwise and full matched signed receipt",
        "by_case": by_case, "fitted_models": fingerprints,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "sklearn": sklearn.__version__},
        "status": "DEVELOPMENT_ONLY_NONBLINDED",
        "limits": [
            "Adversarial explanations are deliberately generated; not external incidents",
            "Ground-truth label permutation is intentionally unobservable without an external label oracle",
            "Manual equal-evidence baseline ties by design; no universal superiority claim",
            "HMAC key in script is demo-only, not secure deployment credential",
            "Signed receipt attests the producer declaration only; compromised producer/key is out of scope",
            "Predictor accuracy, causal faithfulness, and human comprehension are not certified",
            "Correlated mutations of same test case are not independent observations",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/signed_route_development"))
    args = parser.parse_args()
    report = run(args.output_dir)
    print(json.dumps({"models": report["models"], "cases": report["n_records"],
                      "status": report["status"], "by_case": report["by_case"]},
                     ensure_ascii=False, indent=2))