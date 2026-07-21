from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

from fuzzyxai import FuzzyXAI


X, Y = make_classification(n_samples=80, n_features=6, random_state=42)


@pytest.mark.optional_integration
def test_xgboost_native_contributions() -> None:
    xgb = pytest.importorskip("xgboost")
    model = xgb.XGBClassifier(n_estimators=5, max_depth=2, random_state=42).fit(X, Y)
    result = FuzzyXAI.wrap(model).explain_one(X[0], reference_data=X, reference_labels=Y)
    assert result.model_evidence["contribution_method"] == "xgboost_pred_contribs"
    assert np.isfinite(list(result.model_evidence["contributions"].values())).all()


@pytest.mark.optional_integration
def test_lightgbm_native_contributions() -> None:
    lgb = pytest.importorskip("lightgbm")
    model = lgb.LGBMClassifier(n_estimators=5, max_depth=2, random_state=42, verbose=-1).fit(X, Y)
    result = FuzzyXAI.wrap(model).explain_one(X[0], reference_data=X, reference_labels=Y)
    assert result.model_evidence["contribution_method"] == "lightgbm_pred_contrib"
    assert np.isfinite(list(result.model_evidence["contributions"].values())).all()


@pytest.mark.optional_integration
def test_catboost_native_contributions() -> None:
    catboost = pytest.importorskip("catboost")
    model = catboost.CatBoostClassifier(iterations=5, depth=2, random_seed=42, verbose=False).fit(X, Y)
    result = FuzzyXAI.wrap(model).explain_one(X[0], reference_data=X, reference_labels=Y)
    assert result.model_evidence["contribution_method"] == "catboost_shap_values"
    assert np.isfinite(list(result.model_evidence["contributions"].values())).all()
