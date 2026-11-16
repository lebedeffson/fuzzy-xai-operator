"""Opt-in typed F0 composition with explicit, checkable preconditions.

The optional origin record is independent experiment evidence, not an
authenticated signature. Generic compose() keeps its backward-compatible API.
"""
from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_right
from math import isfinite
from typing import Mapping

from .composition import compose
from .diagnostics import DiagnosticState
from .explanation_object import ExplanationObject, Rule
from fuzzyxai.hierarchy.f0 import F0


@dataclass(frozen=True)
class MembershipSpec:
    """Piecewise-linear fuzzy membership on [0,1], including all breakpoints.

    Unlike a 101-point grid, the sup-norm distance between two such
    functions is computed exactly at the union of their knots.
    """
    knots: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if len(self.knots) < 2 or self.knots[0][0] != 0.0 or self.knots[-1][0] != 1.0:
            raise ValueError('membership must span [0,1]')
        for i, (x, y) in enumerate(self.knots):
            if not isfinite(x) or not isfinite(y) or not 0 <= y <= 1:
                raise ValueError('invalid membership knot')
            if i and self.knots[i - 1][0] >= x:
                raise ValueError('membership knots must have increasing x')

    def value(self, x: float) -> float:
        if not 0 <= x <= 1 or not isfinite(x):
            raise ValueError('membership argument outside domain')
        if x == 1.0:
            return float(self.knots[-1][1])
        xs = [p[0] for p in self.knots]
        i = bisect_right(xs, x) - 1
        x0, y0 = self.knots[i]
        x1, y1 = self.knots[i + 1]
        return float(y0 + (y1 - y0) * ((x - x0) / (x1 - x0)))

    def to_json(self) -> list[list[float]]:
        return [[float(x), float(y)] for x, y in self.knots]

    @classmethod
    def from_json(cls, raw: object) -> 'MembershipSpec':
        if (not isinstance(raw, list) or len(raw) < 2
                or any(not isinstance(pair, (list, tuple)) or len(pair) != 2 for pair in raw)):
            raise ValueError('expected at least two [x, membership] breakpoints')
        return cls(tuple((float(x), float(y)) for x, y in raw))

    def distance(self, other: 'MembershipSpec') -> float:
        points = sorted({x for x, _ in self.knots} | {x for x, _ in other.knots})
        return max(abs(self.value(x) - other.value(x)) for x in points)


@dataclass(frozen=True)
class OriginRecord:
    sample_id: str
    model_id: str
    model_version: str
    target_event: str
    probability: float

    def __post_init__(self) -> None:
        if not all((self.sample_id, self.model_id, self.model_version, self.target_event)):
            raise ValueError('origin identity/event is incomplete')
        if not isfinite(self.probability) or not 0 <= self.probability <= 1:
            raise ValueError('invalid origin probability')


@dataclass(frozen=True)
class CompositionPolicy:
    max_score_difference: float = 0.05
    max_membership_distance: float = 0.08
    require_origin_record: bool = True
    verify_membership: bool = True

    def __post_init__(self) -> None:
        for name in ('max_score_difference', 'max_membership_distance'):
            value = getattr(self, name)
            if not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must lie in [0,1]')


def _diag(code: str, msg: str, *, incomplete: bool = False) -> DiagnosticState:
    return DiagnosticState(
        code=code,
        reason=msg,
        severity='warning' if incomplete else 'critical',
        context={'status': 'UNKNOWN' if incomplete else 'REJECT'},
    )


def result_status(value: ExplanationObject | DiagnosticState) -> str:
    if not isinstance(value, DiagnosticState):
        return 'ACCEPT'
    return str(value.context.get('status', 'REJECT'))


def _record_fields(obj: ExplanationObject) -> tuple[str, str, str, str, float, str] | DiagnosticState:
    # The upstream ExplanationObject is mutable after construction. Recheck the
    # physical domain before trusting its uncertainty, loss, rules or metadata.
    if not isinstance(obj.metadata, Mapping):
        return _diag('D_k_incomplete', 'explanation metadata is unavailable', incomplete=True)
    for field in ('uncertainty', 'reduction_loss'):
        try:
            value = float(getattr(obj, field))
        except (TypeError, ValueError, AttributeError):
            return _diag('D_ij', f'invalid {field}')
        if not isfinite(value) or not 0 <= value <= 1:
            return _diag('D_ij', f'{field} outside [0,1]')
    for rule in obj.rules:
        if (not isinstance(rule, Rule) or not isinstance(rule.name, str)
                or not rule.name or not isinstance(rule.conclusion, str)
                or not rule.conclusion or not isinstance(rule.conditions, Mapping)
                or any(not isinstance(k, str) or not isinstance(v, str)
                       for k, v in rule.conditions.items())):
            return _diag('D_k_incomplete', 'invalid rule signature', incomplete=True)
    if not isinstance(obj.activations, Mapping):
        return _diag('D_k_incomplete', 'rule activations are unavailable', incomplete=True)
    for rule_id, activation in obj.activations.items():
        if not isinstance(rule_id, str):
            return _diag('D_ij', 'invalid activation identifier')
        try:
            value = float(activation)
        except (TypeError, ValueError):
            return _diag('D_ij', 'invalid rule activation')
        if not isfinite(value) or not 0 <= value <= 1:
            return _diag('D_ij', 'rule activation outside [0,1]')
    # Metadata has no validation in the upstream ExplanationObject class.
    names = ('sample_id', 'model_id', 'model_version', 'target_event', 'probability', 'scale', 'mapping')
    if any(obj.metadata.get(n) is None or obj.metadata.get(n) == '' for n in names):
        return _diag('D_k_incomplete', 'incomplete explanation interface', incomplete=True)
    t = obj.trace
    if any(not str(getattr(t, k, '')).strip() for k in ('id', 'version', 'timestamp', 'source', 'checksum')):
        return _diag('D_k_trace', 'missing trace field', incomplete=True)
    if obj.metadata['scale'] != 'probability':
        return _diag('D_ij', 'unsupported score scale')
    try:
        score = float(obj.metadata['probability'])
    except (ValueError, TypeError):
        return _diag('D_ij', 'invalid probability type')
    if not isfinite(score) or not 0 <= score <= 1:
        return _diag('D_ij', 'probability outside [0,1]')
    return (
        str(obj.metadata['sample_id']), str(obj.metadata['model_id']),
        str(obj.metadata['model_version']), str(obj.metadata['target_event']),
        score, str(obj.metadata['mapping']),
    )


def _to_event(
    value: tuple[str, str, str, str, float, str],
    canonical_event: str,
) -> float | DiagnosticState:
    *_, event, probability, mapping = value
    if event == canonical_event:
        return probability if mapping == 'identity' else _diag('D_ij', 'invalid event mapping')
    if {event, canonical_event} == {'malignant', 'benign'} and mapping == 'complement':
        return 1.0 - probability
    return _diag('D_ij', 'missing/incorrect event transformation')


def _validate_fuzzy_frame(
    obj: ExplanationObject,
    specs: dict[str, MembershipSpec],
    canonical_event: str,
    score: float,
    policy: CompositionPolicy,
) -> DiagnosticState | None:
    """Verify that numeric event mapping has NOT silently changed fuzzy semantics.

    F0 term_specs and activations are declared in a canonical event frame.
    The numerical complement applies only to probability, not automatically
    to arbitrary fuzzy terms or rule conclusions.
    """
    frame = obj.metadata.get('fuzzy_event')
    rep_term = obj.metadata.get('representation_term')
    if frame is None or rep_term is None:
        return _diag('D_k_incomplete', 'canonical fuzzy frame or representation term missing', incomplete=True)
    if frame != canonical_event:
        return _diag('D_ij', 'fuzzy representation is not in canonical event frame')
    if not isinstance(rep_term, str) or rep_term not in specs:
        return _diag('D_ij', 'representation term is not declared')
    if not isinstance(obj.representation, F0):
        return _diag('D_ij', 'only scalar F0 representation is supported')
    spec = specs[rep_term]
    # A finite probe set CANNOT certify an arbitrary Python callable: a narrow
    # spike between probes would otherwise be silently accepted. Require an
    # explicit binding to the exact piecewise-linear MembershipSpec.value.
    # For a generic F0 callable the available evidence is insufficient.
    fn = obj.representation.mu
    bound_spec = getattr(fn, '__self__', None)
    if not (type(bound_spec) is MembershipSpec
            and getattr(fn, '__func__', None) is MembershipSpec.value
            and bound_spec.knots == spec.knots):
        return _diag('D_k_incomplete', 'F0 callable lacks a verifiable piecewise-linear binding', incomplete=True)
    # Additional deterministic probes verify the declared interface path.
    knots = [x for x, _ in spec.knots]
    probes = knots + [(knots[i] + knots[i+1]) / 2 for i in range(len(knots)-1)]
    if any(abs(obj.representation.membership(x) - spec.value(x)) > 1e-9 for x in probes):
        return _diag('D_ij', 'F0 representation disagrees with declared membership policy')
    for rule in obj.rules:
        if len(rule.conditions) != 1 or 'risk' not in rule.conditions:
            return _diag('D_k_incomplete', 'only unary risk rules are supported', incomplete=True)
        label = rule.conditions['risk']
        if label not in specs:
            return _diag('D_ij', 'rule references undefined risk term')
        alpha = obj.activations.get(rule.name)
        if alpha is None:
            return _diag('D_k_incomplete', 'rule activation is missing', incomplete=True)
        # Small probability drift accepted by policy can produce proportionate
        # activation drift (slope can be steeper than 1).
        xs = [x for x, _ in specs[label].knots]
        slopes = [abs((specs[label].knots[i+1][1]-specs[label].knots[i][1])/
                      (xs[i+1]-xs[i])) for i in range(len(xs)-1)]
        max_drift = min(1.0, max(slopes, default=0.0) * policy.max_score_difference)
        if abs(alpha - specs[label].value(score)) > max(1e-9, max_drift) + 1e-12:
            return _diag('D_ij', 'rule activation contradicts declared fuzzy term and score')
    return None


def compose_checked(
    left: ExplanationObject,
    right: ExplanationObject,
    beta: Mapping[str, float],
    *,
    origin: OriginRecord | None,
    policy: CompositionPolicy = CompositionPolicy(),
) -> ExplanationObject | DiagnosticState:
    """Check typed compatibility, then call the *unaltered* upstream `compose`.

    Conditions tested here are deliberately narrow: binary complementary events
    (malignant/benign), a shared scalar risk domain, exact rule signatures, and
    an F0-compatible distance over the same canonical term.
    """
    l = _record_fields(left)
    if isinstance(l, DiagnosticState):
        return l
    r = _record_fields(right)
    if isinstance(r, DiagnosticState):
        return r
    if origin is None and policy.require_origin_record:
        return _diag('D_k_trace', 'independent origin record unavailable', incomplete=True)
    if l[0] != r[0]:
        return _diag('D_ij', 'sample identifier mismatch')
    if l[1:3] != r[1:3]:
        return _diag('D_ij', 'model identity/version mismatch')
    if origin is not None:
        if l[0] != origin.sample_id or l[1] != origin.model_id or l[2] != origin.model_version:
            return _diag('D_ij', 'provenance differs from independent origin record')
        canonical_event = origin.target_event
    else:
        canonical_event = l[3]
    if canonical_event not in {'malignant', 'benign'}:
        return _diag('D_ij', 'unsupported canonical binary event')
    left_score = _to_event(l, canonical_event)
    right_score = _to_event(r, canonical_event)
    if isinstance(left_score, DiagnosticState):
        return left_score
    if isinstance(right_score, DiagnosticState):
        return right_score
    if abs(left_score - right_score) > policy.max_score_difference + 1e-12:
        return _diag('D_ij', 'cross-module probability mismatch')
    if origin is not None and (
        abs(left_score - origin.probability) > policy.max_score_difference + 1e-12
        or abs(right_score - origin.probability) > policy.max_score_difference + 1e-12
    ):
        return _diag('D_ij', 'probability inconsistent with origin record')
    if not left.terms or left.terms != right.terms:
        return _diag('D_ij', 'term vocabulary mismatch; no declared term mapping')
    signatures: dict[str, str] = {}
    for obj in (left, right):
        for rule in obj.rules:
            signature = rule.signature()
            previous = signatures.setdefault(rule.name, signature)
            if previous != signature:
                return _diag('D_ij', 'same rule id with different signature')
    if policy.verify_membership:
        left_specs = left.metadata.get('term_specs')
        right_specs = right.metadata.get('term_specs')
        if not isinstance(left_specs, dict) or not isinstance(right_specs, dict):
            return _diag('D_k_incomplete', 'membership definitions missing', incomplete=True)
        if set(left_specs) != left.terms or set(right_specs) != right.terms:
            return _diag('D_k_incomplete', 'term definitions do not cover all declared terms', incomplete=True)
        try:
            ls = {name: MembershipSpec.from_json(raw) for name, raw in left_specs.items()}
            rs = {name: MembershipSpec.from_json(raw) for name, raw in right_specs.items()}
        except (ValueError, TypeError, OverflowError):
            return _diag('D_ij', 'invalid piecewise-linear term definition')
        for obj, term_map, score in ((left, ls, left_score), (right, rs, right_score)):
            binding_error = _validate_fuzzy_frame(obj, term_map, canonical_event, score, policy)
            if binding_error is not None:
                return binding_error
        d_mu = max(ls[name].distance(rs[name]) for name in left.terms)
        if d_mu > policy.max_membership_distance + 1e-12:
            return _diag('D_ij', 'membership functions differ beyond declared tolerance')
    result = compose(left, right, beta, allow_missing_terms=False)
    if not isinstance(result, DiagnosticState):
        # Keep the original shared provenance and mapping for downstream checks.
        result.metadata = {
            **result.metadata,
            'sample_id': l[0], 'model_id': l[1], 'model_version': l[2],
            'target_event': canonical_event, 'probability': right_score,
            'mapping': 'identity', 'scale': 'probability',
            'term_specs': right.metadata.get('term_specs'),
            'checked_contract': 'typed-f0-v2',
            'fuzzy_event': canonical_event,
            'representation_term': right.metadata.get('representation_term'),
        }
    return result
