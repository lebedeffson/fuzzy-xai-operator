from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping
import json
from pathlib import Path


def _sum1(weights: Mapping[str, float], eps: float = 1e-9) -> bool:
    return abs(sum(float(v) for v in weights.values()) - 1.0) <= eps


@dataclass
class ExplainPlan:
    """Verifiable specification of interfaces, weights, thresholds, and generated terms.

    The plan can be authored manually or generated from tabular data. Generated
    terms are stored in metadata and can be exported for audit.
    """

    beta: Dict[str, float] = field(default_factory=lambda: {
        'repr': 0.30, 'rules': 0.25, 'activations': 0.15, 'uncertainty': 0.20, 'trace': 0.10
    })
    lambda_: Dict[str, float] = field(default_factory=lambda: {
        'H': 0.20, 'C': 0.20, 'O': 0.20, 'K': 0.20, 'U': 0.20
    })
    eta: Dict[str, float] = field(default_factory=lambda: {
        'model': 0.50, 'rules': 0.30, 'trace': 0.20
    })
    i_min: float = 0.50
    activation_threshold: float = 0.05
    epsilon: float = 1e-3
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not _sum1(self.beta):
            raise ValueError('beta weights must sum to 1')
        if not _sum1(self.lambda_):
            raise ValueError('lambda weights must sum to 1')
        if not _sum1(self.eta):
            raise ValueError('eta weights must sum to 1')
        if not 0 < self.i_min <= 1:
            raise ValueError('i_min must be in (0,1]')
        if not 0 <= self.activation_threshold <= 1:
            raise ValueError('activation_threshold must be in [0,1]')

    def with_reduction_weight(self, beta_delta: float) -> 'ExplainPlan':
        if not 0 <= beta_delta < 1:
            raise ValueError('beta_delta must be in [0,1)')
        beta = {k: (1 - beta_delta) * v for k, v in self.beta.items()}
        beta['reduction'] = beta_delta
        return ExplainPlan(beta=beta, lambda_=dict(self.lambda_), eta=dict(self.eta), i_min=self.i_min,
                           activation_threshold=self.activation_threshold, epsilon=self.epsilon,
                           metadata=dict(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        return {
            'beta': dict(self.beta),
            'lambda': dict(self.lambda_),
            'eta': dict(self.eta),
            'i_min': self.i_min,
            'activation_threshold': self.activation_threshold,
            'epsilon': self.epsilon,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> 'ExplainPlan':
        plan = cls()
        if 'beta' in data:
            plan.beta = dict(data['beta'])
        if 'lambda' in data or 'lambda_' in data:
            plan.lambda_ = dict(data.get('lambda', data.get('lambda_', {})))
        if 'eta' in data:
            plan.eta = dict(data['eta'])
        plan.i_min = float(data.get('i_min', plan.i_min))
        plan.activation_threshold = float(data.get('activation_threshold', plan.activation_threshold))
        plan.epsilon = float(data.get('epsilon', plan.epsilon))
        plan.metadata = dict(data.get('metadata', {}))
        plan.validate()
        return plan

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')

    @classmethod
    def load_json(cls, path: str | Path) -> 'ExplainPlan':
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))

    @classmethod
    def from_data(cls, X, y=None, *, target=None, n_terms: int = 3, mode: str = 'audit') -> 'ExplainPlan':
        from .plan_builder import build_explain_plan_from_dataframe
        return build_explain_plan_from_dataframe(X, target=target, n_terms=n_terms, mode=mode)
