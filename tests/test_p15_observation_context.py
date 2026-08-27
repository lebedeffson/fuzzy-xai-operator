"""P15.6: ObservationContext — one comprehensive registration point.

Before this, a caller who wanted a fully-populated result had to separately
call observe_training(), remember to pass its output back into
explain_one(training_run=..., include_training_trace=True), and repeat
reference_data/reference_labels on every call — with nothing tying "the
data", "the trained model", and "the final explanation" together. This
verifies: (1) data-checking (observe_training) still works completely
standalone; (2) explaining a ready model still works completely standalone
with no context registered; (3) registering an ObservationContext makes one
explain_one() call comprehensive without repeating anything.
"""

from __future__ import annotations

from fuzzyxai import FuzzyXAI
from fuzzyxai.runtime import ObservationContext
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def _split():
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)


def test_data_observation_works_standalone_without_a_context() -> None:
    X_train, _, y_train, _ = _split()
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    fx = FuzzyXAI.wrap(model)
    history = {"objects": {"p0": [{"epoch": 1, "correct": True, "confidence": 0.9}, {"epoch": 2, "correct": False, "confidence": 0.4}]}}
    analysis = fx.observe_training(history=history)
    assert "p0" in analysis.traces


def test_model_explanation_works_standalone_without_a_context() -> None:
    X_train, X_test, y_train, _ = _split()
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    fx = FuzzyXAI.wrap(model)
    result = fx.explain_one(X_test[0], object_id="p0")
    assert result.prediction.predictions is not None
    assert result.similar_cases == ()  # no reference corpus registered, none fabricated


def test_registered_context_combines_reference_corpus_and_training_into_one_call() -> None:
    X_train, X_test, y_train, _ = _split()
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    train_ids = [f"train_{i}" for i in range(len(X_train))]

    probe = FuzzyXAI.wrap(model)
    history = {"objects": {"p0": [{"epoch": 1, "correct": True, "confidence": 0.9}, {"epoch": 2, "correct": False, "confidence": 0.4}]}}
    training_run = probe.observe_training(history=history)

    context = ObservationContext(
        reference_data=X_train,
        reference_labels=y_train,
        reference_ids=train_ids,
        training_run=training_run,
        dataset_version="breast_cancer_v1",
    )
    fx = FuzzyXAI.wrap(model, observation_context=context)
    result = fx.explain_one(X_test[0], object_id="p0", include_counterfactuals=True, include_training_trace=True)

    # P16/P17/P18: a bare LogisticRegression has exactly one local-explanation
    # channel, so Γ (alignment) has no genuine second explanatory object to
    # compare against automatically, and there is no real representation-
    # reduction operation (Π) for it either — its default ExplainPlan never
    # declares either applicable, so both are honestly `not_applicable`, not
    # "missing" (this scenario never called for them). E4 is the correct,
    # honest ceiling here, not a fabricated E5. Everything ELSE this one call
    # combines (reference corpus, training history, dataset metadata,
    # counterfactuals) is still fully present.
    assert result.explanation_level == "E4"
    assert set(result.missing_channels) == set()
    assert {"alignment", "reduction"} <= set(result.view_model.explanation_level["not_applicable_channels"])
    assert len(result.similar_cases) > 0
    assert result.view_model.trace["dataset_version"] == "breast_cancer_v1"
    assert result.view_model.layers.get("training")


def test_explicit_per_call_arguments_still_override_the_context() -> None:
    X_train, X_test, y_train, _ = _split()
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    context = ObservationContext(reference_data=X_train, reference_labels=y_train, dataset_version="from_context")
    fx = FuzzyXAI.wrap(model, observation_context=context)
    result = fx.explain_one(X_test[0], object_id="p0", dataset_version="from_call")
    assert result.view_model.trace["dataset_version"] == "from_call"


def test_explicit_reference_data_kwarg_wins_over_context_reference_data() -> None:
    X_train, X_test, y_train, _ = _split()
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    # Deliberately different (smaller) reference set passed directly to wrap().
    context = ObservationContext(reference_data=X_train[:5], reference_labels=y_train[:5])
    fx = FuzzyXAI.wrap(model, reference_data=X_train, reference_labels=y_train, observation_context=context)
    result = fx.explain_one(X_test[0], object_id="p0")
    top = result.similar_cases[0]
    assert top["reference_count"] == len(X_train)  # used the explicit kwarg's full set, not the context's 5-row subset


def test_context_without_training_run_does_not_fabricate_training_evidence() -> None:
    X_train, X_test, y_train, _ = _split()
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    context = ObservationContext(reference_data=X_train, reference_labels=y_train)  # no training_run
    fx = FuzzyXAI.wrap(model, observation_context=context)
    result = fx.explain_one(X_test[0], object_id="p0")
    assert "training_history" in result.missing_channels
    assert not result.view_model.layers.get("training")
