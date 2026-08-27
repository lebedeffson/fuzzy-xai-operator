from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, cast

import yaml


def _sum1(weights: Mapping[str, float], eps: float = 1e-9) -> bool:
    return abs(sum(float(v) for v in weights.values()) - 1.0) <= eps


@dataclass(frozen=True)
class MembershipTerm:
    """One labeled fuzzy term of a MembershipPolicy's variable — e.g. label
    'medium', function 'triangular', parameters (a, b, c)."""

    label: str
    function: str
    parameters: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {'label': self.label, 'function': self.function, 'parameters': list(self.parameters)}

    def membership(self, x: float) -> float:
        """Evaluate a declared triangular term, including shoulder endpoints."""
        if self.function != "triangular" or len(self.parameters) != 3:
            raise ValueError(f"unsupported membership function: {self.function}")
        a, b, c = self.parameters
        value = float(x)
        if not a <= value <= c:
            return 0.0
        if (a == b and value == a) or (b == c and value == c) or value == b:
            return 1.0
        if value < b:
            return (value - a) / (b - a)
        return (c - value) / (c - b)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MembershipTerm:
        return cls(label=str(data['label']), function=str(data['function']), parameters=tuple(float(v) for v in data['parameters']))


@dataclass(frozen=True)
class MembershipPolicy:
    """P16 section 17: the single, disclosed source of a variable's fuzzy
    membership functions — no unexplained triangles. ``origin`` records
    whether the parameters came from a domain expert, a calibration
    procedure, a fixed preset, or an imported/legacy source; ``version``
    and ``calibration_reference`` make the policy traceable to whatever
    produced it.
    """

    variable: str
    universe: tuple[float, float]
    terms: tuple[MembershipTerm, ...]
    origin: str
    version: str
    calibration_reference: str | None = None

    def __post_init__(self) -> None:
        if self.origin not in {'expert', 'calibrated', 'preset', 'imported'}:
            raise ValueError("MembershipPolicy.origin must be 'expert', 'calibrated', 'preset', or 'imported'")
        if not self.terms:
            raise ValueError('MembershipPolicy requires at least one term')

    def to_dict(self) -> dict[str, Any]:
        return {
            'variable': self.variable,
            'universe': list(self.universe),
            'terms': [term.to_dict() for term in self.terms],
            'origin': self.origin,
            'version': self.version,
            'calibration_reference': self.calibration_reference,
        }

    def evaluate(self, x: float) -> dict[str, float]:
        return {term.label: term.membership(float(x)) for term in self.terms}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MembershipPolicy:
        return cls(
            variable=str(data['variable']),
            universe=tuple(float(v) for v in data['universe']),  # type: ignore[arg-type]
            terms=tuple(MembershipTerm.from_dict(item) for item in data['terms']),
            origin=str(data['origin']),
            version=str(data['version']),
            calibration_reference=data.get('calibration_reference'),
        )


_UNCERTAINTY_METHODS = {
    "none",
    "entropy",
    "margin",
    "ensemble_disagreement",
    "ensemble_vote_standard_deviation",
    "calibrated_interval",
}


@dataclass(frozen=True)
class UncertaintyPolicy:
    """P18: the single, disclosed source of u_M / U_model.

    ``method`` names which real signal the risk observer is allowed to read
    as ``uncertainty`` for objects explained under this plan:

    - ``"none"``: this scenario has no declared uncertainty source. The
      ``uncertainty`` risk component is then genuinely not applicable (not
      "missing") for objects explained under this plan.
    - ``"entropy"``: normalized Shannon entropy of the model's own class
      probability vector.
    - ``"margin"``: ``1 - |p_top - p_second|`` from the model's own class
      probability vector (a real predictive-margin uncertainty, matching the
      pattern used by the chapter 5 reference demo's ``evaluate_vector``).
    - ``"ensemble_vote_standard_deviation"``: standard deviation of binary
      per-estimator votes. Its range is [0, 0.5].
    - ``"ensemble_disagreement"``: backward-compatible alias for
      ``"ensemble_vote_standard_deviation"``; it is not a variance.
    - ``"calibrated_interval"``: half-width of a calibrated prediction
      interval, when the adapter genuinely supplies one.

    Declaring a method other than ``"none"`` and then genuinely lacking the
    corresponding signal is a real ``missing_required`` condition -- unlike
    ``"none"``, which means the component was never expected in the first
    place. ``surrogate_fidelity_gap`` (how well a local surrogate matches the
    black box) is never a valid source here -- it answers a different
    question (explanation fidelity, not predictive uncertainty).
    """

    method: str = "none"
    parameters: Dict[str, Any] = field(default_factory=dict)
    source: str = ""

    def __post_init__(self) -> None:
        if self.method not in _UNCERTAINTY_METHODS:
            raise ValueError(f"UncertaintyPolicy.method must be one of {sorted(_UNCERTAINTY_METHODS)}")

    @property
    def applicable(self) -> bool:
        return self.method != "none"

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "parameters": dict(self.parameters), "source": self.source}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> UncertaintyPolicy:
        return cls(method=str(data.get("method", "none")), parameters=dict(data.get("parameters", {})), source=str(data.get("source", "")))


@dataclass(frozen=True)
class ReductionPolicy:
    """P18: whether a real representation-reduction operation (Pi) is part
    of this plan's scenario at all.

    Most model families in this framework have no automatic Pi -- ``Delta``
    (reduction loss) is then correctly ``not_applied``, never
    ``missing_required``. ``applicable=True`` declares that this scenario
    genuinely performs a measured reduction (the caller supplies
    ``evidence={"reduction": ...}``, or a future adapter computes one
    in-process); only then does its absence become a real gap.
    """

    applicable: bool = False
    method: str = "none"
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"applicable": self.applicable, "method": self.method, "source": self.source}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReductionPolicy:
        return cls(applicable=bool(data.get("applicable", False)), method=str(data.get("method", "none")), source=str(data.get("source", "")))


@dataclass(frozen=True)
class AlignmentPolicy:
    """P18: whether a real second explanatory channel (E_i -> E_j) is
    expected to exist for this scenario, i.e. whether Gamma is a genuine
    part of the route rather than an opportunistic extra.

    A single-channel model (most sklearn linear/tree/ensemble adapters) has
    ``applicable=False`` by default: Gamma stays ``not_applicable`` for it,
    never ``missing_required``, and it is never penalized for lacking a
    second channel it was never supposed to have.
    """

    applicable: bool = False
    source: str = ""
    transform: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"applicable": self.applicable, "source": self.source, "transform": dict(self.transform)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AlignmentPolicy:
        return cls(
            applicable=bool(data.get("applicable", False)),
            source=str(data.get("source", "")),
            transform=dict(data.get("transform", {})),
        )


@dataclass(frozen=True)
class UncertaintyRepresentationPolicy:
    """Declared construction of the uncertainty representation used by Pi.

    The P19 validation preset is a disclosed heuristic interval, not a
    calibrated prediction interval: ``[p - scale*U_model, p + scale*U_model]``
    clipped to ``clip``.
    """

    method: str = "vote_probability_plus_minus_dispersion"
    scale: float = 1.0
    clip: tuple[float, float] = (0.0, 1.0)
    source: str = "ExplainPlan P19 validation preset"

    def __post_init__(self) -> None:
        if self.method not in {"none", "vote_probability_plus_minus_dispersion"}:
            raise ValueError("unsupported uncertainty representation policy")
        if self.scale < 0:
            raise ValueError("uncertainty representation scale must be non-negative")
        if len(self.clip) != 2 or self.clip[0] >= self.clip[1]:
            raise ValueError("uncertainty representation clip must be an ordered pair")

    @property
    def applicable(self) -> bool:
        return self.method != "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "scale": self.scale,
            "clip": list(self.clip),
            "source": self.source,
            "calibrated": False,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> UncertaintyRepresentationPolicy:
        return cls(
            method=str(data.get("method", "vote_probability_plus_minus_dispersion")),
            scale=float(data.get("scale", 1.0)),
            clip=tuple(float(value) for value in data.get("clip", (0.0, 1.0))),  # type: ignore[arg-type]
            source=str(data.get("source", "")),
        )


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

    domain_language = plan.get('domain_language', {})
    if not isinstance(domain_language, Mapping):
        raise ValueError('ExplainPlan.domain_language must be a mapping')
    for section in ('features', 'classes', 'actions'):
        values = domain_language.get(section, {})
        if not isinstance(values, Mapping):
            raise ValueError(f'ExplainPlan.domain_language.{section} must be a mapping')


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
    domain_language: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    membership_policies: Dict[str, MembershipPolicy] = field(default_factory=dict)
    uncertainty_policy: UncertaintyPolicy = field(default_factory=UncertaintyPolicy)
    reduction_policy: ReductionPolicy = field(default_factory=ReductionPolicy)
    alignment_policy: AlignmentPolicy = field(default_factory=AlignmentPolicy)
    uncertainty_representation_policy: UncertaintyRepresentationPolicy = field(default_factory=UncertaintyRepresentationPolicy)

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
        if not isinstance(self.domain_language, dict):
            raise ValueError('domain_language must be a mapping')
        for section in ('features', 'classes', 'actions'):
            if not isinstance(self.domain_language.get(section, {}), Mapping):
                raise ValueError(f'domain_language.{section} must be a mapping')

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
            domain_language=dict(self.domain_language),
            metadata=dict(self.metadata),
            membership_policies=dict(self.membership_policies),
            uncertainty_policy=self.uncertainty_policy,
            reduction_policy=self.reduction_policy,
            alignment_policy=self.alignment_policy,
            uncertainty_representation_policy=self.uncertainty_representation_policy,
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
            'domain_language': self.domain_language,
            'metadata': self.metadata,
            'membership_policies': {name: policy.to_dict() for name, policy in self.membership_policies.items()},
            'uncertainty_policy': self.uncertainty_policy.to_dict(),
            'reduction_policy': self.reduction_policy.to_dict(),
            'alignment_policy': self.alignment_policy.to_dict(),
            'uncertainty_representation_policy': self.uncertainty_representation_policy.to_dict(),
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
        plan.domain_language = dict(data.get('domain_language', {}))
        plan.metadata = dict(data.get('metadata', {}))
        plan.membership_policies = {
            str(name): (policy if isinstance(policy, MembershipPolicy) else MembershipPolicy.from_dict(policy))
            for name, policy in data.get('membership_policies', {}).items()
        }
        if 'uncertainty_policy' in data:
            raw = data['uncertainty_policy']
            plan.uncertainty_policy = raw if isinstance(raw, UncertaintyPolicy) else UncertaintyPolicy.from_dict(raw)
        if 'reduction_policy' in data:
            raw = data['reduction_policy']
            plan.reduction_policy = raw if isinstance(raw, ReductionPolicy) else ReductionPolicy.from_dict(raw)
        if 'alignment_policy' in data:
            raw = data['alignment_policy']
            plan.alignment_policy = raw if isinstance(raw, AlignmentPolicy) else AlignmentPolicy.from_dict(raw)
        if 'uncertainty_representation_policy' in data:
            raw = data['uncertainty_representation_policy']
            plan.uncertainty_representation_policy = (
                raw if isinstance(raw, UncertaintyRepresentationPolicy)
                else UncertaintyRepresentationPolicy.from_dict(raw)
            )
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
    def from_data(
        cls,
        X: Any,
        y: Any | None = None,
        *,
        target: Any | None = None,
        n_terms: int = 3,
        mode: str = 'audit',
    ) -> 'ExplainPlan':
        from .plan_builder import build_explain_plan_from_dataframe
        return cast(ExplainPlan, build_explain_plan_from_dataframe(X, target=target, n_terms=n_terms, mode=mode))
