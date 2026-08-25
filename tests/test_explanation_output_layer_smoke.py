"""Cross-model/cross-task smoke checks for the explanation output layer.

Per the project's evidence-first scope, this deliberately does not attempt to
measure or claim human comprehension (see docs/human_explanation_layer.md and
context/RESEARCH.md's "Current Scientific Position"). It only asserts that
raw-object highlighting and verbalization run end-to-end, without exception,
across more than one task/model combination, and that evidence-first
invariants hold (an unmapped feature is reported, never silently dropped).
"""

from __future__ import annotations

import pytest
from fuzzyxai import FuzzyXAI
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def _fit_logistic_regression():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    return model, X_test


def _assert_object_representation_well_formed(result) -> None:
    spec = result.view_model.visual_spec
    representation = spec.get("object_representation")
    assert representation is not None, "raw_objects was supplied; object_representation must not be silently absent"
    assert representation["modality"] == "text"
    assert isinstance(representation["highlighted_html"], str) and representation["highlighted_html"]
    assert isinstance(representation["unmapped_features"], (list, tuple))

    figure = result.visualize(view="object_representation", backend="matplotlib")
    assert figure is not None
    figure_plotly = result.visualize(view="object_representation", backend="plotly")
    assert figure_plotly is not None

    verbalized = result.verbalize()
    assert isinstance(verbalized, str) and verbalized


def test_tabular_logistic_regression_output_layer_runs() -> None:
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    fx = FuzzyXAI.wrap(model, adapter="auto", task="classification")
    result = fx.explain_one(
        X_test[0],
        object_id="tabular-lr-0",
        raw_object="feature_0 and feature_1 were both elevated in this scan.",
    )
    _assert_object_representation_well_formed(result)


def test_tabular_random_forest_without_contributions_stays_evidence_first() -> None:
    """RandomForest via the generic adapter yields no local contributions today
    (a pre-existing framework property, not something this feature changes).
    The output layer must not fabricate a contribution or a text-highlight
    match in that case. It still gets a tabular object representation (raw
    feature values, per the tabular fallback), but every row's contribution
    is honestly None/"unknown" rather than invented — this test locks in
    that disclosure rather than either silence or fabrication.
    """

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    model = RandomForestClassifier(n_estimators=25, random_state=0).fit(X_train, y_train)
    fx = FuzzyXAI.wrap(model, adapter="auto", task="classification")
    result = fx.explain_one(
        X_test[0],
        object_id="tabular-rf-0",
        raw_object="Notes mention feature_2 as the dominant signal.",
    )
    assert result.view_model.model.get("contributions") == {}
    assert "text_highlight_contributions" in result.view_model.trace.get("missing_evidence", [])

    representation = result.view_model.visual_spec.get("object_representation")
    assert representation is not None
    assert representation["modality"] == "tabular"
    assert representation["tabular_rows"], "tabular fallback should still show raw feature values"
    assert all(row["contribution"] is None and row["direction"] == "unknown" for row in representation["tabular_rows"])


def test_text_tfidf_linear_classifier_output_layer_runs() -> None:
    documents = [
        "the router keeps dropping the wifi connection at night",
        "wifi signal is unstable and drops every evening",
        "invoice payment was declined by the billing system",
        "billing system rejected the invoice payment again",
    ]
    labels = ["network", "network", "billing", "billing"]
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(documents).toarray()
    model = LogisticRegression(max_iter=2000).fit(features, labels)
    feature_names = list(vectorizer.get_feature_names_out())

    fx = FuzzyXAI.wrap(model, adapter="auto", task="classification")
    query_text = "wifi connection drops constantly"
    query_vector = vectorizer.transform([query_text]).toarray()[0]
    result = fx.explain_one(
        query_vector,
        object_id="text-tfidf-0",
        feature_names=feature_names,
        raw_object=query_text,
    )
    _assert_object_representation_well_formed(result)
    representation = result.view_model.visual_spec["object_representation"]
    matched = {span["feature_name"] for span in representation["spans"]}
    assert matched, "at least one TF-IDF token should be lexically findable in its own query text"


def test_no_raw_object_falls_back_to_tabular_default() -> None:
    model, X_test = _fit_logistic_regression()
    fx = FuzzyXAI.wrap(model, adapter="auto", task="classification")
    result = fx.explain_one(X_test[0], object_id="no-raw-0")
    representation = result.view_model.visual_spec.get("object_representation")
    assert representation is not None, "omitting raw_object must not silently produce no representation at all"
    assert representation["modality"] == "tabular"
    assert representation["tabular_rows"]
    assert any(row["contribution"] is not None for row in representation["tabular_rows"])


def test_unsupported_raw_object_type_does_not_fabricate_text_evidence() -> None:
    model, X_test = _fit_logistic_regression()
    fx = FuzzyXAI.wrap(model, adapter="auto", task="classification")
    result = fx.explain_one(X_test[0], object_id="unsupported-0", raw_object=12345)
    assert "text_highlight_unsupported_raw_object_type" in result.view_model.trace.get("missing_evidence", [])
    representation = result.view_model.visual_spec.get("object_representation")
    # No text spans were guessed for the unsupported int; the tabular
    # fallback (built from independent, already-collected data evidence)
    # still gives an honest representation instead of leaving it empty.
    assert representation is not None
    assert representation["modality"] == "tabular"
    assert len(representation["spans"]) == 0


def test_raw_objects_cardinality_mismatch_raises_value_error() -> None:
    model, X_test = _fit_logistic_regression()
    fx = FuzzyXAI.wrap(model, adapter="auto", task="classification")
    with pytest.raises(ValueError, match="raw_objects"):
        fx.explain_batch(
            X_test[:2],
            object_ids=["a", "b"],
            raw_objects=["only one raw object for two rows"],
        )
