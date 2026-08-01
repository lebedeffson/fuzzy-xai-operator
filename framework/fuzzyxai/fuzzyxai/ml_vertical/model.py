from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd
import shap
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fuzzyxai.diagnostics.contracts import canonical_sha256

from .contracts import LocalExplanationArtifact, PredictionArtifact

SEED = 1729
MODEL_ID = "bcw-logistic-regression"
MODEL_VERSION = "1.0.0"
EXPECTED_SCHEMA_SHA256 = "5feb56528d0e0bc2070783a206139d9900013b692d15ffdf8c3c0a4127e48aba"


class BreastCancerModel:
    def __init__(self) -> None:
        dataset = load_breast_cancer(as_frame=True)
        self.feature_names = tuple(str(name) for name in dataset.feature_names)
        self.schema_sha256 = canonical_sha256(self.feature_names)
        if self.schema_sha256 != EXPECTED_SCHEMA_SHA256:
            raise RuntimeError("registered feature schema does not match sklearn dataset")
        x_train_val, self.x_test, y_train_val, self.y_test = train_test_split(
            dataset.data, dataset.target, test_size=0.2, random_state=SEED, stratify=dataset.target,
        )
        self.x_train, self.x_validation, self.y_train, self.y_validation = train_test_split(
            x_train_val, y_train_val, test_size=0.2, random_state=SEED, stratify=y_train_val,
        )
        self.scaler = StandardScaler().fit(self.x_train)
        # The software demonstration defines class 1 as the elevated (malignant) score.
        train_target = 1 - self.y_train.to_numpy()
        self.model = LogisticRegression(max_iter=2000, random_state=SEED).fit(
            self.scaler.transform(self.x_train), train_target,
        )
        self.explainer = shap.LinearExplainer(self.model, self.scaler.transform(self.x_train))
        self.model_sha256 = canonical_sha256(
            {
                "coef": self.model.coef_.tolist(),
                "intercept": self.model.intercept_.tolist(),
                "scale": self.scaler.scale_.tolist(),
                "mean": self.scaler.mean_.tolist(),
                "sklearn": sklearn.__version__,
            }
        )

    @cached_property
    def default_features(self) -> dict[str, float]:
        row = self.x_test.iloc[0]
        return {name: float(row[name]) for name in self.feature_names}

    def validate_features(self, features: dict[str, float]) -> tuple[str, ...]:
        return tuple(name for name in self.feature_names if name not in features)

    def vector(self, features: dict[str, float]) -> pd.DataFrame:
        return pd.DataFrame([[float(features[name]) for name in self.feature_names]], columns=self.feature_names)

    def predict(self, object_id: str, features: dict[str, float]) -> PredictionArtifact:
        transformed = self.scaler.transform(self.vector(features))
        probability = float(self.model.predict_proba(transformed)[0, 1])
        raw_score = float(self.model.decision_function(transformed)[0])
        return PredictionArtifact(
            object_id, int(probability >= 0.5), probability, raw_score,
            MODEL_ID, MODEL_VERSION, self.model_sha256, self.schema_sha256,
        )

    def explain(
        self,
        object_id: str,
        features: dict[str, float],
        *,
        observed_model_version: str = MODEL_VERSION,
    ) -> LocalExplanationArtifact:
        transformed = self.scaler.transform(self.vector(features))
        result = self.explainer(transformed)
        values = np.asarray(result.values[0], dtype=float)
        base = float(np.asarray(result.base_values).reshape(-1)[0])
        output_sum = base + float(values.sum())
        raw_score = float(self.model.decision_function(transformed)[0])
        payload: dict[str, Any] = {
            "object_id": object_id,
            "explainer_id": "shap.LinearExplainer",
            "explainer_version": shap.__version__,
            "model_version": observed_model_version,
            "base_value": base,
            "shap_values": dict(zip(self.feature_names, values.tolist(), strict=True)),
            "feature_values": features,
            "output_sum": output_sum,
            "output_difference": abs(output_sum - raw_score),
        }
        payload["artifact_sha256"] = canonical_sha256(payload)
        return LocalExplanationArtifact(**payload)

    def manifest(self) -> dict[str, Any]:
        return {
            "model_id": MODEL_ID,
            "model_version": MODEL_VERSION,
            "model_sha256": self.model_sha256,
            "feature_schema_sha256": self.schema_sha256,
            "feature_names": self.feature_names,
            "sklearn_version": sklearn.__version__,
            "shap_version": shap.__version__,
        }
