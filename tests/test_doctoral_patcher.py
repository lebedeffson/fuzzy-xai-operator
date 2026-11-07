"""Patcher regression: preflight is read-only, edits are syntactic and idempotent."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import apply_doctoral_upgrade as patcher


def source_snapshots() -> dict[str, str]:
    runtime = '''from __future__ import annotations
from typing import Any, Mapping
class ModelExplanationResult:
    def diagnose(
        self,
        *,
        repair_mode: str = "plan",
        repair_context: RepairExecutionContext | None = None,
        audience: str = "user",
    ) -> DiagnosticReport:
        return None
class FuzzyXAI:
    def explain_one(self, input_object: Any, **kwargs: Any):
        return None
    def explain_batch(
        self,
        inputs: Any,
        *,
        object_ids: list[str] | None = None,
    ) -> ModelExplanationResult:
        return None
'''
    risk = '''from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
CANONICAL_RISK_COMPONENTS = ("rho_p","u_M","one_minus_I_pre","Delta","chi_R")
def compute_risk(components: Mapping[str, float], weights: Mapping[str, float]) -> float:
    return 0.0
def compute_canonical_risk(components: Mapping[str,float], weights: Mapping[str,float]) -> float:
    if any(float(value) < 0 for value in weights.values()):
        raise ValueError("canonical rho weights must be non-negative")
    if abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-9:
        raise ValueError("canonical rho weights must sum to one; implicit renormalization is forbidden")
    return sum(float(weights[key]) * max(0.0, min(1.0, float(components[key]))) for key in CANONICAL_RISK_COMPONENTS)
def _candidate_action(rho, thresholds):
    theta_1 = float(thresholds["theta_1"])
    return "accept"
def compute_action(rho,chi_r_crit,thresholds):
    candidate = "accept"
    return "block" if int(chi_r_crit) == 1 else candidate
def observe_risk(rho,chi_r_crit):
    critical = int(chi_r_crit)
    return critical
def observe_legacy_normalized_risk(rho,chi_r_crit):
    critical = int(chi_r_crit)
    return critical
'''
    alignment = '''from __future__ import annotations
from dataclasses import dataclass
def compute_gamma_route(p,alpha_mean,feature_support):
    gamma = abs(float(p) - float(feature_support))
    return round(gamma, 6)
'''
    application = '''from __future__ import annotations
from dataclasses import dataclass
import numpy as np
DEFAULT_RISK_WEIGHTS = {"a":0.5,"b":0.5}
def clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
def normalize_risk_weights(weights=None):
    merged = dict(DEFAULT_RISK_WEIGHTS)
    if weights:
        merged.update({str(k): max(0.0, float(v)) for k, v in weights.items()})
    if any(value < 0.0 for value in merged.values()):
        raise ValueError("risk weights must be non-negative")
    return merged
def expected_action_costs(proba,cost_matrix):
    p = np.asarray(proba, dtype=float).reshape(-1)
    total = float(p.sum())
    if total <= 0.0:
        raise ValueError('proba must have positive sum')
    p = p / total
    costs = {}
    for action, row in cost_matrix.items():
        c = np.asarray(row,dtype=float).reshape(-1)
        if c.size != p.size:
            raise ValueError(f'cost row for {action!r} has length {c.size}, expected {p.size}')
        costs[str(action)] = float(np.dot(p, c))
    return costs
'''
    return {
        'framework/fuzzyxai/fuzzyxai/runtime.py': runtime,
        'framework/fuzzyxai/fuzzyxai/__init__.py': '__all__ = []\n',
        'framework/fuzzyxai/fuzzyxai/core/risk_observer.py': risk,
        'framework/fuzzyxai/fuzzyxai/risk/risk_function.py': application,
        'framework/fuzzyxai/fuzzyxai/core/alignment.py': alignment,
        'pyproject.toml': '[project.scripts]\nfuzzyxai = "fuzzyxai.cli:main"\n',
    }


def test_preflight_no_mutation_and_idempotent(tmp_path, monkeypatch):
    sources = source_snapshots()
    for name, contents in sources.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(contents, encoding='utf-8')
    monkeypatch.setattr(patcher,'ROOT',tmp_path)
    monkeypatch.setattr(sys,'argv',['apply_doctoral_upgrade','--check'])
    patcher.main()
    assert all((tmp_path/name).read_text(encoding='utf-8')==src for name,src in sources.items())
    monkeypatch.setattr(sys,'argv',['apply_doctoral_upgrade'])
    patcher.main()
    first = {name:(tmp_path/name).read_text(encoding='utf-8') for name in sources}
    assert all(first[name]!=src for name,src in sources.items())
    for name,text in first.items():
        if name.endswith('.py'):
            ast.parse(text)
    patcher.main()
    assert all((tmp_path/name).read_text(encoding='utf-8')==src for name,src in first.items())
    assert 'def explain_one_certified(' in first['framework/fuzzyxai/fuzzyxai/runtime.py']
    assert '_finite_risk_number(' in first['framework/fuzzyxai/fuzzyxai/risk/risk_function.py']


def test_preflight_stops_before_any_write_if_one_target_is_missing(tmp_path, monkeypatch):
    sources = source_snapshots()
    for name, contents in sources.items():
        if name=='pyproject.toml': continue
        p=tmp_path/name
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(contents,encoding='utf-8')
    monkeypatch.setattr(patcher,'ROOT',tmp_path)
    monkeypatch.setattr(sys,'argv',['apply_doctoral_upgrade'])
    with pytest.raises(FileNotFoundError):
        patcher.main()
    assert all((tmp_path/name).read_text(encoding='utf-8')==src for name,src in sources.items() if name!='pyproject.toml')


def test_repaired_core_functions_execute_with_numeric_safety():
    sources=source_snapshots()
    path='framework/fuzzyxai/fuzzyxai/core/risk_observer.py'
    code=patcher.patch_risk(sources[path])
    namespace={}
    exec(compile(code,path,'exec'),namespace)
    weights={'rho_p':.3,'u_M':.2,'one_minus_I_pre':.2,'Delta':.2,'chi_R':.1}
    comps={k:.2 for k in weights}
    assert namespace['compute_canonical_risk'](comps,weights)==pytest.approx(.2)
    with pytest.raises(ValueError,match='finite'):
        namespace['compute_canonical_risk']({**comps,'u_M':float('nan')},weights)
    with pytest.raises(ValueError,match='exactly'):
        namespace['_binary_override'](2)
    assert namespace['_binary_override'](True)==1

    alignment=patcher.patch_alignment(sources['framework/fuzzyxai/fuzzyxai/core/alignment.py'])
    region={}
    exec(compile(alignment,'alignment.py','exec'),region)
    assert region['compute_gamma_route'](.67,.31,.60)==pytest.approx(.36)
    with pytest.raises(ValueError):
        region['compute_gamma_route'](.67,float('inf'),.60)

    application=patcher.patch_application_risk(sources['framework/fuzzyxai/fuzzyxai/risk/risk_function.py'])
    ap={}
    exec(compile(application,'risk_function.py','exec'),ap)
    with pytest.raises(ValueError,match='finite'):
        ap['clip01'](float('nan'))
    with pytest.raises(ValueError,match='non-negative'):
        ap['normalize_risk_weights']({'a':-.3})
    with pytest.raises(ValueError,match='unknown'):
        ap['normalize_risk_weights']({'invalid':0.0})
    with pytest.raises(ValueError,match='finite'):
        ap['expected_action_costs']([float('inf'),.5], {'accept':[0,1]})
