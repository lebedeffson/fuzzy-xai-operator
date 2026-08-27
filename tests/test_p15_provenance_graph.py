"""P15.14: explicit lifecycle provenance nodes (dataset, preprocessor,
model_artifact) and a transitive `inspect()` that surfaces the whole chain
from a claim back to its dataset, matching the user's worked example:
dataset -> preprocessor -> transformed value -> coefficient ->
local contribution -> claim -> prediction.

Split/checkpoint nodes are intentionally NOT added — the framework has no
tracking infrastructure for those yet, and fabricating them would violate
evidence-first.
"""

from __future__ import annotations

import pandas as pd
from fuzzyxai import FuzzyXAI
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _pipeline_result(dataset_version: str | None = "breast_cancer_v1"):
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=3000))]).fit(X, y)
    kwargs = {} if dataset_version is None else {"dataset_version": dataset_version}
    return FuzzyXAI.wrap(pipe).explain_one(X.iloc[0].to_numpy(), object_id="p0", feature_names=list(X.columns), **kwargs)


def test_dataset_and_model_artifact_nodes_are_present() -> None:
    result = _pipeline_result()
    node_types = {node["node_type"] for node in result.view_model.explanation_graph["nodes"]}
    assert {"dataset", "model_artifact", "preprocessor"} <= node_types


def test_graph_stays_reachable_with_provenance_nodes() -> None:
    result = _pipeline_result()
    assert result.explanation_graph.validate_reachability() == ()


def test_inspect_walks_the_full_chain_from_claim_to_dataset() -> None:
    """The user's exact worked example: dataset -> preprocessor ->
    transformed value -> coefficient -> local contribution -> claim ->
    prediction, all from one `inspect("claim:...")` call."""

    result = _pipeline_result()
    linear_claim = next(c for c in result.claims if c.claim_type == "linear_reconstruction")
    inspection = result.inspect(f"claim:{linear_claim.claim_id}")
    node_types = {node.node_type for node in inspection.related_nodes}
    assert {"dataset", "preprocessor", "model_internals", "claim", "prediction"} <= node_types


def test_no_preprocessor_node_for_a_bare_non_pipeline_model() -> None:
    from sklearn.datasets import load_breast_cancer as load

    X, y = load(return_X_y=True)
    model = LogisticRegression(max_iter=3000).fit(X, y)
    result = FuzzyXAI.wrap(model).explain_one(X[0], object_id="p0")
    node_types = {node["node_type"] for node in result.view_model.explanation_graph["nodes"]}
    assert "preprocessor" not in node_types


def test_no_dataset_node_fabricated_when_dataset_version_absent() -> None:
    """dataset_version defaults to 'unversioned' by runtime.py, so a dataset
    node is still created honestly labeled as unversioned — but nothing is
    invented beyond what the caller actually told the framework."""

    result = _pipeline_result(dataset_version=None)
    dataset_nodes = [node for node in result.view_model.explanation_graph["nodes"] if node["node_type"] == "dataset"]
    assert len(dataset_nodes) == 1
    assert dataset_nodes[0]["payload"]["dataset_version"] == "unversioned"
