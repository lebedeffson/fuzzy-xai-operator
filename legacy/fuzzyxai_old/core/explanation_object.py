from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Set

@dataclass(frozen=True)
class Trace:
    """Traceable provenance tuple tau_k."""
    id: str
    version: str
    timestamp: str
    params: Mapping[str, Any] = field(default_factory=dict)
    source: str = ''
    checksum: str = ''

    def as_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'version': self.version,
            'timestamp': self.timestamp,
            'params': dict(self.params),
            'source': self.source,
            'checksum': self.checksum,
        }

@dataclass(frozen=True)
class Rule:
    """Interpretable rule."""
    name: str
    conditions: Mapping[str, str]
    conclusion: str

    def signature(self) -> str:
        body = '|'.join(f'{k}={v}' for k, v in sorted(self.conditions.items()))
        return f'{self.name}|{body}|{self.conclusion}'

@dataclass
class ExplanationObject:
    """Fuzzy explanation object E_k.

    representation is A_k^F. It may be F0, IntervalFS, HesitantFS,
    NeutrosophicFS, or MultiLevelFS.
    """
    terms: Set[str]
    representation: Any
    rules: List[Rule]
    activations: Dict[str, float]
    uncertainty: float
    trace: Trace
    reduction_loss: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 <= self.uncertainty <= 1:
            raise ValueError('uncertainty must be in [0,1]')
        if not 0 <= self.reduction_loss <= 1:
            raise ValueError('reduction_loss must be in [0,1]')
        for name, value in self.activations.items():
            if not 0 <= float(value) <= 1:
                raise ValueError(f'activation {name} must be in [0,1]')

    @property
    def active_rules(self) -> Set[str]:
        threshold = float(self.metadata.get('activation_threshold', 0.05))
        return {r.name for r in self.rules if self.activations.get(r.name, 0.0) > threshold}

    def copy_with(self, **kwargs: Any) -> 'ExplanationObject':
        data = {
            'terms': set(self.terms),
            'representation': self.representation,
            'rules': list(self.rules),
            'activations': dict(self.activations),
            'uncertainty': self.uncertainty,
            'trace': self.trace,
            'reduction_loss': self.reduction_loss,
            'metadata': dict(self.metadata),
        }
        data.update(kwargs)
        return ExplanationObject(**data)
