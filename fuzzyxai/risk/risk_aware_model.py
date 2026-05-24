from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

import numpy as np

from fuzzyxai import ExplainPlan, Rule, SystemOperator, Trace, interpretability_index, interpretability_loss

from .decision_policy import RiskPolicy
from .uncertainty import confidence_from_uncertainty, entropy_uncertainty, margin_uncertainty


class RiskAwareModel:
    """Decision-gate wrapper around a fitted or trainable probabilistic model."""

    def __init__(
        self,
        base_model,
        plan: ExplainPlan | None = None,
        policy: RiskPolicy | None = None,
        positive_class: int | str = 1,
        mode: str = 'decision_gate',
        cost_matrix: Any = None,
    ) -> None:
        self.base_model = base_model
        self.plan = plan
        self.policy = policy or RiskPolicy()
        self.positive_class = positive_class
        self.mode = mode
        self.cost_matrix = cost_matrix
        self._operator: SystemOperator | None = SystemOperator(plan) if plan is not None else None

    def fit(self, X, y):
        self.base_model.fit(X, y)
        if self.plan is None:
            self.plan = ExplainPlan.from_data(X, y, mode='audit').with_reduction_weight(0.10)
            self._operator = SystemOperator(self.plan)
        return self

    def predict_proba(self, X):
        if hasattr(self.base_model, 'predict_proba'):
            return self.base_model.predict_proba(X)
        if hasattr(self.base_model, 'decision_function'):
            scores = np.asarray(self.base_model.decision_function(X), dtype=float)
            if scores.ndim == 1:
                p1 = 1.0 / (1.0 + np.exp(-scores))
                return np.column_stack([1.0 - p1, p1])
        pred = np.asarray(self.base_model.predict(X), dtype=float)
        return np.column_stack([1.0 - pred, pred])

    def predict(self, X):
        return self.base_model.predict(X)

    def _risk_column(self, proba: np.ndarray) -> int:
        classes = getattr(self.base_model, 'classes_', None)
        if classes is not None:
            for idx, label in enumerate(classes):
                if label == self.positive_class:
                    return idx
        return min(1, proba.shape[1] - 1)

    def predict_with_risk(self, X, metadata: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        if self.plan is None or self._operator is None:
            raise ValueError('RiskAwareModel must be fitted or initialized with an ExplainPlan')
        metadata = dict(metadata or {})
        proba = np.asarray(self.predict_proba(X), dtype=float)
        raw_pred = np.asarray(self.predict(X))
        risk_col = self._risk_column(proba)
        risks = proba[:, risk_col]
        ent = entropy_uncertainty(proba)
        margin = margin_uncertainty(proba)
        uncertainty = np.clip(0.70 * ent + 0.30 * margin, 0.0, 1.0)
        confidence = confidence_from_uncertainty(uncertainty)
        rules = [
            Rule('risk_high', {'risk': 'high'}, 'defer_or_review'),
            Rule('risk_medium', {'risk': 'medium'}, 'lower_confidence'),
        ]
        outputs: list[dict[str, Any]] = []
        for idx, risk in enumerate(risks):
            diagnostics = list(metadata.get('diagnostics', []) or [])
            if metadata.get('force_diagnostic', False):
                diagnostics.append('D_ij')
            explanation = self._operator.explain_scalar_risk(
                float(risk),
                rules,
                Trace(f'risk-aware-model-{idx}', 'v1', 'runtime', source=metadata.get('source', 'risk-aware-model'), checksum='risk-aware'),
                model_uncertainty=float(uncertainty[idx]),
                trace_uncertainty=0.01,
            )
            loss = interpretability_loss(0.30, 0.33, 0.16, 0.03, explanation.uncertainty, self.plan.lambda_, explanation.reduction_loss, 0.10)
            pre_index = interpretability_index(loss)
            decision = self.policy.choose(float(risk), float(uncertainty[idx]), float(pre_index), float(explanation.reduction_loss), diagnostics)
            outputs.append({
                'raw_prediction': raw_pred[idx].item() if hasattr(raw_pred[idx], 'item') else raw_pred[idx],
                'raw_proba': [float(v) for v in proba[idx]],
                'predicted_risk': float(risk),
                'uncertainty': float(uncertainty[idx]),
                'entropy_uncertainty': float(ent[idx]),
                'margin_uncertainty': float(margin[idx]),
                'confidence': float(confidence[idx]),
                'selected_representation': getattr(explanation.representation, 'class_name', type(explanation.representation).__name__),
                'reduction_loss': float(explanation.reduction_loss),
                'pre_interpretability': float(pre_index),
                'final_interpretability': float(pre_index),
                'interpretability_index': float(pre_index),
                'application_risk': float(decision.risk_score),
                'risk_score': float(decision.risk_score),
                'action': decision.action.value,
                'corrected_confidence': float(decision.corrected_confidence),
                'reason': decision.reason,
                'diagnostics': list(decision.diagnostics),
                'explanation_object': explanation,
                'risk_decision': asdict(decision),
            })
        return outputs

    def evaluate(self, X, y) -> dict[str, Any]:
        from .metrics import accepted_accuracy, block_rate, coverage, defer_rate, mean_interpretability

        outputs = self.predict_with_risk(X)
        y_pred = [row['raw_prediction'] for row in outputs]
        actions = [row['action'] for row in outputs]
        indices = [row['interpretability_index'] for row in outputs]
        return {
            'coverage': coverage(actions),
            'accepted_accuracy': accepted_accuracy(y, y_pred, actions),
            'defer_rate': defer_rate(actions),
            'block_rate': block_rate(actions),
            'mean_interpretability': mean_interpretability(indices),
        }
