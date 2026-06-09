from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fuzzyxai_experiments.src.utils import read_yaml_or_json, sha256_text


@dataclass(frozen=True)
class ExplainPlan:
    """YAML/JSON ExplainPlan wrapper with deterministic hashing."""
    path: str
    data: dict[str, Any]

    @classmethod
    def load(cls, path: str) -> 'ExplainPlan':
        """Load a plan from YAML or JSON."""
        return cls(path=path, data=read_yaml_or_json(path))

    def hash(self) -> str:
        """Return SHA256 of canonical plan serialization."""
        import json
        return sha256_text(json.dumps(self.data, ensure_ascii=False, sort_keys=True, separators=(',', ':')))

    def beta(self) -> dict[str, float]:
        """Return composition weights."""
        return {k: float(v) for k, v in self.data.get('composition', {}).get('beta', {}).items()}


@dataclass(frozen=True)
class Rule:
    """Simple rule R_k with condition and conclusion."""
    rule_id: str
    condition: str
    conclusion: str

    def signature(self) -> str:
        """Return stable rule signature."""
        return f'{self.rule_id}:{self.condition}->{self.conclusion}'


@dataclass(frozen=True)
class Trace:
    """Trace tuple tau_k."""
    source: str
    version: str
    timestamp: str
    params: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize trace."""
        return {'source': self.source, 'version': self.version, 'timestamp': self.timestamp, 'params': self.params}


@dataclass
class DiagnosticState:
    """Diagnostic object D_ij for failed composition/synthesis."""
    code: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize diagnostic."""
        return {'code': self.code, 'details': self.details}


@dataclass
class FuzzyExplanation:
    """Object E_k = (L, mu, R, alpha, u, tau)."""
    L: set[str]
    mu: dict[str, float]
    R: list[Rule]
    alpha: dict[str, float]
    u: float
    tau: Trace
    reduction_loss: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize explanation into JSON-compatible fields."""
        return {
            'L': sorted(self.L),
            'mu': dict(self.mu),
            'R': [r.signature() for r in self.R],
            'alpha': dict(self.alpha),
            'u': self.u,
            'tau': self.tau.as_dict(),
            'reduction_loss': self.reduction_loss,
            'metadata': dict(self.metadata),
        }

    def compute_d_E(self, other: 'FuzzyExplanation', plan: ExplainPlan) -> float:
        """Compute weighted semantic disagreement d_E."""
        beta = plan.beta() or {'mu': 0.25, 'R': 0.2, 'alpha': 0.2, 'u': 0.2, 'tau': 0.15}
        mu_keys = set(self.mu) | set(other.mu)
        mu_gap = sum(abs(self.mu.get(k, 0.0) - other.mu.get(k, 0.0)) for k in mu_keys) / max(1, len(mu_keys))
        rule_gap = 0.0 if {r.signature() for r in self.R} == {r.signature() for r in other.R} else 1.0
        alpha_keys = set(self.alpha) | set(other.alpha)
        alpha_gap = sum(abs(self.alpha.get(k, 0.0) - other.alpha.get(k, 0.0)) for k in alpha_keys) / max(1, len(alpha_keys))
        u_gap = abs(self.u - other.u)
        tau_gap = 0.0 if self.tau.source == other.tau.source else 1.0
        return round(beta.get('mu', 0.25) * mu_gap + beta.get('R', 0.2) * rule_gap + beta.get('alpha', 0.2) * alpha_gap + beta.get('u', 0.2) * u_gap + beta.get('tau', 0.15) * tau_gap, 6)

    def compute_L(self, plan: ExplainPlan) -> float:
        """Compute simple interpretability loss L(E)."""
        weights = plan.data.get('interpretability', {}).get('lambda', {'H': .2, 'C': .2, 'O': .2, 'K': .2, 'U': .2})
        return round(float(weights.get('U', .2)) * self.u + self.reduction_loss, 6)

    def uncertainty(self) -> float:
        """Return uncertainty u."""
        return float(self.u)


def compose(E_i: FuzzyExplanation, E_j: FuzzyExplanation, T_ij: dict[str, Any], plan: ExplainPlan) -> FuzzyExplanation | DiagnosticState:
    """Compose two explanations or return diagnostic D_ij."""
    gamma = E_i.compute_d_E(E_j, plan)
    gamma_max = float(T_ij.get('gamma_max', plan.data.get('composition', {}).get('gamma_max', 0.45)))
    if gamma > gamma_max:
        return DiagnosticState('composition_disagreement', {'gamma': gamma, 'gamma_max': gamma_max})
    return FuzzyExplanation(
        L=E_i.L | E_j.L,
        mu={**E_i.mu, **E_j.mu},
        R=[*E_i.R, *E_j.R],
        alpha={**E_i.alpha, **E_j.alpha},
        u=max(E_i.u, E_j.u),
        tau=E_j.tau,
        reduction_loss=max(E_i.reduction_loss, E_j.reduction_loss),
        metadata={'gamma': gamma, 'composed_from': [E_i.tau.source, E_j.tau.source]},
    )


def synthesize_alignment(E_i: FuzzyExplanation, E_j: FuzzyExplanation, plan: ExplainPlan) -> dict[str, Any] | DiagnosticState:
    """Run limited alignment synthesis T_ij over finite candidates."""
    cfg = plan.data.get('alignment_synthesis', {})
    candidates = cfg.get('candidates')
    if not candidates:
        return DiagnosticState('alignment_space_not_finite', {'required': 'finite candidates in ExplainPlan'})
    gamma_max = float(cfg.get('gamma_max', plan.data.get('composition', {}).get('gamma_max', 0.45)))
    lambda_delta = float(cfg.get('lambda_delta', 0.2))
    gamma = E_i.compute_d_E(E_j, plan)
    scored = []
    for cand in candidates:
        delta_t = float(cand.get('Delta_T', 0.0))
        j_value = gamma + lambda_delta * delta_t
        scored.append({'candidate': cand.get('name', 'candidate'), 'gamma': gamma, 'Delta_T': delta_t, 'J(T)': j_value})
    best = min(scored, key=lambda row: row['J(T)'])
    if best['J(T)'] > gamma_max:
        return DiagnosticState('alignment_cost_too_high', {'best': best, 'gamma_max': gamma_max})
    return best
