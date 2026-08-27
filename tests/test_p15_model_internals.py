"""P15.1/P15.3: model-internals evidence is surfaced, not discarded.

Before this, `SklearnLinearAdapter`/`SklearnTreeAdapter`/`SklearnEnsembleAdapter`
already computed rich model-specific evidence in `extract_local_evidence`
(coefficients + intercept, the literal tree decision path + leaf, ensemble
votes + disagreement + global importance) — but `runtime.py::explain()` only
ever read `contributions`/`contribution_method` out of that payload and
silently dropped the rest. This verifies each model family's real internals
now reach the final result via `evidence.model_internals` and, where they add
genuinely new information (decision path, ensemble votes), a corresponding
human-facing claim.
"""

from __future__ import annotations

from fuzzyxai import FuzzyXAI
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def _split():
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)


def test_linear_model_surfaces_coefficients_and_intercept() -> None:
    X_train, X_test, y_train, _ = _split()
    model = LogisticRegression(max_iter=3000).fit(X_train, y_train)
    result = FuzzyXAI.wrap(model).explain_one(X_test[0], object_id="p0")
    internals = result.view_model.layers["model_internals"]
    assert len(internals) == 1
    assert internals[0]["model_family"] == "sklearn_linear"
    assert internals[0]["coefficients"]
    assert internals[0]["intercept"] is not None
    assert internals[0]["decision_path"] is None  # a linear model genuinely has no decision path


def test_tree_model_surfaces_decision_path_and_leaf() -> None:
    X_train, X_test, y_train, _ = _split()
    model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_train, y_train)
    result = FuzzyXAI.wrap(model).explain_one(X_test[0], object_id="p0")
    internals = result.view_model.layers["model_internals"][0]
    assert internals["decision_path"]
    assert internals["leaf_id"] is not None
    assert internals["leaf_samples"] is not None
    assert internals["coefficients"] is None  # a tree genuinely has no coefficients
    claim_types = {claim["claim_type"] for claim in result.view_model.claims}
    assert "decision_path" in claim_types


def test_ensemble_model_surfaces_votes_and_disagreement() -> None:
    X_train, X_test, y_train, _ = _split()
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X_train, y_train)
    result = FuzzyXAI.wrap(model).explain_one(X_test[0], object_id="p0")
    internals = result.view_model.layers["model_internals"][0]
    assert internals["ensemble_votes"]
    assert internals["ensemble_disagreement"] is not None
    assert internals["global_importance"]
    claim_types = {claim["claim_type"] for claim in result.view_model.claims}
    assert "ensemble_votes" in claim_types


def test_model_internals_claims_are_reachable_in_the_graph() -> None:
    X_train, X_test, y_train, _ = _split()
    model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_train, y_train)
    result = FuzzyXAI.wrap(model).explain_one(X_test[0], object_id="p0")
    assert result.explanation_graph.validate_reachability() == ()


def test_linear_pipeline_shows_raw_to_transformed_reconstruction_chain() -> None:
    """The user's exact worked example: a StandardScaler->LogisticRegression
    pipeline should surface, per feature, raw_value -> transformed_value ->
    coefficient -> contribution, plus a reconstructed_score that matches the
    model's own decision_function (reconstruction_error ~ 0)."""

    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=3000))]).fit(X_train, y_train)

    result = FuzzyXAI.wrap(pipe).explain_one(X_test.iloc[0].to_numpy(), object_id="p0", feature_names=list(X.columns))
    internals = result.view_model.layers["model_internals"][0]
    assert internals["pipeline_steps"] == ["scaler", "clf"]
    term = internals["linear_terms"][0]
    assert term["raw_value"] is not None and term["transformed_value"] is not None
    assert term["raw_value"] != term["transformed_value"]
    assert internals["reconstruction_error"] < 1e-6


def test_full_summary_narrates_the_actual_tree_path_not_all_rules() -> None:
    """P15.3: `summary(detail='full')` must narrate the object's own path
    through the tree, not a dump of every rule extracted from the tree."""

    X_train, X_test, y_train, _ = _split()
    model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_train, y_train)
    result = FuzzyXAI.wrap(model).explain_one(X_test[0], object_id="p0")
    he = result.explain_for("user")
    titles = [reason.title for reason in he.details.supports]
    assert "Путь по дереву решений" in titles


def test_full_summary_narrates_ensemble_vote_counts() -> None:
    X_train, X_test, y_train, _ = _split()
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X_train, y_train)
    result = FuzzyXAI.wrap(model).explain_one(X_test[0], object_id="p0")
    he = result.explain_for("user")
    titles = [reason.title for reason in he.details.supports]
    assert "Голосование ансамбля" in titles


def test_linear_reconstruction_chain_appears_in_technical_metrics() -> None:
    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=3000))]).fit(X_train, y_train)
    result = FuzzyXAI.wrap(pipe).explain_one(X_test.iloc[0].to_numpy(), object_id="p0", feature_names=list(X.columns))
    he = result.explain_for("user")
    titles = [item.title for item in he.details.technical_metrics]
    assert "Разложение линейной оценки" in titles


def test_model_internals_absence_is_disclosed_not_fabricated() -> None:
    from fuzzyxai.adapters.contracts_v2 import ExplanationContext, LocalModelEvidence
    from fuzzyxai.adapters.model import ModelPrediction
    from fuzzyxai.adapters.model_v2 import ModelAdapterV2

    class BareAdapter(ModelAdapterV2):
        adapter_id = "bare"
        model_family = "bare"

        def predict(self, inputs):
            return ModelPrediction(predictions=[0], probabilities=None, model_type="bare", adapter_id=self.adapter_id, metadata={"task_type": self.task_type.value})

        def extract_local_evidence(self, inputs, prediction, context: ExplanationContext) -> LocalModelEvidence:
            return LocalModelEvidence(channels={})

        def feature_names(self):
            return ["a"]

        def model_fingerprint(self):
            return "0" * 16

    result = FuzzyXAI.wrap(object(), adapter=BareAdapter(object(), task="classification")).explain_one([1.0], object_id="p0")
    assert result.view_model.layers["model_internals"] == []
    assert "model_internals" in result.view_model.layers["missing"]
