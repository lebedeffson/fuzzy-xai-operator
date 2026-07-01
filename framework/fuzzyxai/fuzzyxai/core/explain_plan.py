from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping
import hashlib
import json
from pathlib import Path

import yaml


def _sum1(weights: Mapping[str, float], eps: float = 1e-9) -> bool:
    return abs(sum(float(v) for v in weights.values()) - 1.0) <= eps


def load_explain_plan(path: str | Path) -> dict[str, Any]:
    """Load a YAML or JSON ExplainPlan contract."""
    p = Path(path)
    text = p.read_text(encoding='utf-8')
    if p.suffix.lower() == '.json':
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError('ExplainPlan file must contain a mapping')
    return data


def _require_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f'ExplainPlan.{key} must be a mapping')
    return value


def _require_sequence(data: Mapping[str, Any], key: str) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f'ExplainPlan.{key} must be a non-empty list')
    return value


def validate_explain_plan(plan: Mapping[str, Any]) -> None:
    """Validate the machine-readable ExplainPlan contract used by reports."""
    if not plan.get('version'):
        raise ValueError('ExplainPlan.version is required')
    if not plan.get('name'):
        raise ValueError('ExplainPlan.name is required')

    terms = _require_mapping(plan, 'terms')
    risk_terms = _require_mapping(terms, 'risk')
    labels = _require_sequence(risk_terms, 'L')
    membership = _require_mapping(risk_terms, 'membership')
    nodes = _require_mapping(membership, 'nodes')
    for label in labels:
        tri = nodes.get(str(label))
        if not isinstance(tri, list) or len(tri) != 3:
            raise ValueError(f'membership node for {label!r} must have 3 values')
        a, b, c = [float(v) for v in tri]
        if not (0.0 <= a <= b <= c <= 1.0):
            raise ValueError(f'membership node for {label!r} must be ordered in [0,1]')

    rules = _require_sequence(plan, 'rules')
    for rule in rules:
        if not isinstance(rule, Mapping) or not rule.get('id') or 'if' not in rule or 'then' not in rule:
            raise ValueError('each rule must contain id/if/then')

    uncertainty = _require_mapping(plan, 'uncertainty')
    eta = _require_mapping(uncertainty, 'weights')
    if not _sum1({str(k): float(v) for k, v in eta.items()}):
        raise ValueError('uncertainty.weights must sum to 1')

    trace_required = _require_sequence(plan, 'trace_required')
    for field_name in ('id', 'version', 'time', 'params', 'source', 'hash'):
        if field_name not in trace_required:
            raise ValueError(f'trace_required must contain {field_name}')

    composition = _require_mapping(plan, 'composition')
    beta = _require_mapping(composition, 'beta')
    if not _sum1({str(k): float(v) for k, v in beta.items()}):
        raise ValueError('composition.beta must sum to 1')

    risk_observer = _require_mapping(plan, 'risk_observer')
    weights = _require_mapping(risk_observer, 'weights')
    if not _sum1({str(k): float(v) for k, v in weights.items()}):
        raise ValueError('risk_observer.weights must sum to 1')
    thresholds = risk_observer.get('thresholds')
    if not isinstance(thresholds, list) or len(thresholds) != 4:
        raise ValueError('risk_observer.thresholds must contain 4 values')
    t1, t2, t3, t4 = [float(v) for v in thresholds]
    if not (0.0 <= t1 < t2 < t3 < t4 <= 1.0):
        raise ValueError('risk_observer.thresholds must be ordered in [0,1]')


def canonicalize_explain_plan(plan: Mapping[str, Any]) -> str:
    """Deterministic JSON serialization used as the hash source."""
    validate_explain_plan(plan)
    return json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def hash_explain_plan(plan: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonicalize_explain_plan(plan).encode('utf-8')).hexdigest()


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
    gamma_warning: float = 0.25
    gamma_critical: float = 0.60
    delta_warning: float = 0.25
    delta_critical: float = 0.60
    rho_accept: float = 0.35
    rho_warning: float = 0.60
    rho_audit: float = 0.75
    rho_critical: float = 0.85
    top_k: int = 5
    representation_policy: str = "auto"
    action_policy: str = "risk_zone"
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
        if not (0 <= self.gamma_warning <= self.gamma_critical <= 1):
            raise ValueError('gamma thresholds must be ordered in [0,1]')
        if not (0 <= self.delta_warning <= self.delta_critical <= 1):
            raise ValueError('delta thresholds must be ordered in [0,1]')
        if not (0 <= self.rho_accept <= self.rho_warning <= self.rho_audit <= self.rho_critical <= 1):
            raise ValueError('rho thresholds must be ordered in [0,1]')
        if self.top_k <= 0:
            raise ValueError('top_k must be positive')

    def with_reduction_weight(self, beta_delta: float) -> 'ExplainPlan':
        if not 0 <= beta_delta < 1:
            raise ValueError('beta_delta must be in [0,1)')
        beta = {k: (1 - beta_delta) * v for k, v in self.beta.items()}
        beta['reduction'] = beta_delta
        return ExplainPlan(
            beta=beta,
            lambda_=dict(self.lambda_),
            eta=dict(self.eta),
            i_min=self.i_min,
            activation_threshold=self.activation_threshold,
            epsilon=self.epsilon,
            gamma_warning=self.gamma_warning,
            gamma_critical=self.gamma_critical,
            delta_warning=self.delta_warning,
            delta_critical=self.delta_critical,
            rho_accept=self.rho_accept,
            rho_warning=self.rho_warning,
            rho_audit=self.rho_audit,
            rho_critical=self.rho_critical,
            top_k=self.top_k,
            representation_policy=self.representation_policy,
            action_policy=self.action_policy,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'beta': dict(self.beta),
            'lambda': dict(self.lambda_),
            'eta': dict(self.eta),
            'i_min': self.i_min,
            'activation_threshold': self.activation_threshold,
            'epsilon': self.epsilon,
            'gamma_warning': self.gamma_warning,
            'gamma_critical': self.gamma_critical,
            'delta_warning': self.delta_warning,
            'delta_critical': self.delta_critical,
            'rho_accept': self.rho_accept,
            'rho_warning': self.rho_warning,
            'rho_audit': self.rho_audit,
            'rho_critical': self.rho_critical,
            'top_k': self.top_k,
            'representation_policy': self.representation_policy,
            'action_policy': self.action_policy,
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
        for key in (
            'gamma_warning', 'gamma_critical', 'delta_warning', 'delta_critical',
            'rho_accept', 'rho_warning', 'rho_audit', 'rho_critical',
        ):
            if key in data:
                setattr(plan, key, float(data[key]))
        if 'top_k' in data:
            plan.top_k = int(data['top_k'])
        plan.representation_policy = str(data.get('representation_policy', plan.representation_policy))
        plan.action_policy = str(data.get('action_policy', plan.action_policy))
        plan.metadata = dict(data.get('metadata', {}))
        plan.validate()
        return plan

    @classmethod
    def default(cls) -> 'ExplainPlan':
        return cls()

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')

    @classmethod
    def load_json(cls, path: str | Path) -> 'ExplainPlan':
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))

    @classmethod
    def from_data(cls, X, y=None, *, target=None, n_terms: int = 3, mode: str = 'audit') -> 'ExplainPlan':
        from .plan_builder import build_explain_plan_from_dataframe
        return build_explain_plan_from_dataframe(X, target=target, n_terms=n_terms, mode=mode)
