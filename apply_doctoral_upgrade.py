"""Apply narrowly-scoped chapter-1-to-4 integration to a clean source clone.

New files from this overlay should be copied to the repository first.
All existing-file edits are validated before writing; --check never writes.
Edits are idempotent, and can be run again after copying a newer overlay.
"""
from __future__ import annotations

import argparse
import ast
import math
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def replace_once(text: str, needle: str, replacement: str, label: str) -> str:
    count = text.count(needle)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, observed {count}; inspect upstream source")
    return text.replace(needle, replacement, 1)


def patch_runtime(text: str) -> str:
    if "def explain_one_certified(" in text:
        if "def certify(\n        self,\n        *,\n        taxonomy:" not in text:
            raise RuntimeError("runtime: certified facade present but result.certify missing")
        return text
    result_anchor = '''    def diagnose(
        self,
        *,
        repair_mode: str = "plan",
        repair_context: RepairExecutionContext | None = None,
        audience: str = "user",
    ) -> DiagnosticReport:
'''
    result_method = '''    def certify(
        self,
        *,
        taxonomy: Mapping[str, Any],
        representation: Mapping[str, Any],
        policy: Mapping[str, Any] | None = None,
        repair_costs: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        """Verify chapter 1-4 evidence contracts (not prediction correctness)."""
        from fuzzyxai.doctoral_certification import certify_result

        return certify_result(self, taxonomy=taxonomy, representation=representation,
                              policy=policy, repair_costs=repair_costs)

'''
    before_model = text.split("class FuzzyXAI:")[0]
    if before_model.count(result_anchor) != 1:
        raise RuntimeError("runtime: ModelExplanationResult.diagnose anchor changed")
    text = replace_once(text, result_anchor, result_method + result_anchor, "runtime.certify")
    facade_anchor = '''    def explain_batch(
        self,
        inputs: Any,
        *,
        object_ids: list[str] | None = None,
'''
    facade_method = '''    def explain_one_certified(
        self,
        input_object: Any,
        *,
        taxonomy: Mapping[str, Any],
        representation: Mapping[str, Any],
        certification_policy: Mapping[str, Any] | None = None,
        **explain_kwargs: Any,
    ) -> tuple[ModelExplanationResult, dict[str, Any]]:
        """Explain and independently certify the same object/route.

        A certificate with status!=valid MUST NOT be treated as permission
        to automate the prediction. The certificate is structural evidence.
        """
        result = self.explain_one(input_object, **explain_kwargs)
        certificate = result.certify(taxonomy=taxonomy, representation=representation,
                                     policy=certification_policy)
        return result, certificate

'''
    return replace_once(text, facade_anchor, facade_method + facade_anchor, "runtime.explain_one_certified")


def patch_init(text: str) -> str:
    imports = (
        "from .doctoral_certification import certify_result, certify_payload, verify_certificate, project_for_audience, select_representation, fuzzy_and\n"
        "from .doctoral_batch import certify_batch\n"
        "from .doctoral_report import render_doctoral_report\n"
    )
    if "from .doctoral_certification import certify_result" in text:
        if "from .doctoral_report import render_doctoral_report" not in text:
            raise RuntimeError("partial doctoral exports already applied; inspect manually")
        return text
    declarations = ('__all__ += ["certify_result", "certify_payload", "verify_certificate", '
                    '"project_for_audience", "select_representation", "fuzzy_and", '
                    '"certify_batch", "render_doctoral_report"]\n')
    return text.rstrip() + "\n\n" + imports + declarations


def patch_risk(text: str) -> str:
    if "def _finite_number(value:" in text:
        return text
    text = replace_once(text, "from dataclasses import dataclass\n", "from dataclasses import dataclass\nimport math\n", "risk math import")
    helper = '''def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _binary_override(value: object) -> int:
    if value is True or value is False:
        return int(value)
    if value not in (0, 1, 0.0, 1.0):
        raise ValueError("chi_r_crit must be exactly 0 or 1")
    return int(value)


'''
    text = replace_once(text, "def compute_risk(components: Mapping[str, float], weights: Mapping[str, float]) -> float:\n", helper + "def compute_risk(components: Mapping[str, float], weights: Mapping[str, float]) -> float:\n", "risk helper")
    text = replace_once(text,
       '''    if any(float(value) < 0 for value in weights.values()):
        raise ValueError("canonical rho weights must be non-negative")
    if abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-9:
        raise ValueError("canonical rho weights must sum to one; implicit renormalization is forbidden")
    return sum(float(weights[key]) * max(0.0, min(1.0, float(components[key]))) for key in CANONICAL_RISK_COMPONENTS)
''',
       '''    checked_weights = {key: _finite_number(weights[key], f"weight:{key}") for key in CANONICAL_RISK_COMPONENTS}
    if any(value < 0 for value in checked_weights.values()):
        raise ValueError("canonical rho weights must be non-negative")
    if abs(sum(checked_weights.values()) - 1.0) > 1e-9:
        raise ValueError("canonical rho weights must sum to one; implicit renormalization is forbidden")
    checked_components = {key: _finite_number(components[key], f"risk:{key}") for key in CANONICAL_RISK_COMPONENTS}
    return sum(checked_weights[key] * max(0.0, min(1.0, checked_components[key])) for key in CANONICAL_RISK_COMPONENTS)
''', "risk finite arithmetic")
    text = replace_once(text, '''    theta_1 = float(thresholds["theta_1"])
''', '''    _finite_number(rho, "rho")
    theta_1 = float(thresholds["theta_1"])
''', "risk finite rho")
    text = replace_once(text,
       '''    return "block" if int(chi_r_crit) == 1 else candidate
''',
       '''    return "block" if _binary_override(chi_r_crit) == 1 else candidate
''', "risk critical override")
    if text.count("critical = int(chi_r_crit)") != 2:
        raise RuntimeError("risk critical override paths changed")
    text = text.replace("critical = int(chi_r_crit)", "critical = _binary_override(chi_r_crit)")
    return text


def patch_application_risk(text: str) -> str:
    """Reject NaN/Inf/negative/unknown weights at the public risk boundary."""
    if "def _finite_risk_number(" in text:
        return text
    text = replace_once(text, "from dataclasses import dataclass\n", "from dataclasses import dataclass\nimport math\n", "application risk math import")
    text = replace_once(text,
        "def clip01(value: float) -> float:\n    return max(0.0, min(1.0, float(value)))\n",
        '''def _finite_risk_number(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite numeric value")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite numeric value") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def clip01(value: float) -> float:
    checked = _finite_risk_number(value, "risk component")
    return max(0.0, min(1.0, checked))
''', "application risk finite input")
    text = replace_once(text,
        '''    if weights:
        merged.update({str(k): max(0.0, float(v)) for k, v in weights.items()})
    if any(value < 0.0 for value in merged.values()):
        raise ValueError("risk weights must be non-negative")
''',
        '''    if weights:
        unknown = set(weights) - set(DEFAULT_RISK_WEIGHTS)
        if unknown:
            raise ValueError(f"unknown risk component weights: {sorted(unknown)}")
        merged.update({str(k): _finite_risk_number(v, f"weight:{k}") for k, v in weights.items()})
    if any(value < 0.0 for value in merged.values()):
        raise ValueError("risk weights must be non-negative")
''', "application risk weight validation")
    text = replace_once(text,
        '''    total = float(p.sum())
    if total <= 0.0:
        raise ValueError('proba must have positive sum')
''',
        '''    if not np.all(np.isfinite(p)) or np.any(p < 0.0):
        raise ValueError('proba must contain finite non-negative probabilities')
    total = float(p.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError('proba must have finite positive sum')
''', "expected cost probability guard")
    text = replace_once(text,
        '''        if c.size != p.size:
            raise ValueError(f'cost row for {action!r} has length {c.size}, expected {p.size}')
        costs[str(action)] = float(np.dot(p, c))
''',
        '''        if c.size != p.size:
            raise ValueError(f'cost row for {action!r} has length {c.size}, expected {p.size}')
        if not np.all(np.isfinite(c)):
            raise ValueError(f'cost row for {action!r} contains non-finite values')
        costs[str(action)] = float(np.dot(p, c))
''', "expected cost finite guard")
    return text


def patch_alignment(text: str) -> str:
    if "max(abs(float(p) - float(alpha_mean)), abs(float(p) - float(feature_support)))" in text:
        return text
    old='''    gamma = abs(float(p) - float(feature_support))
    return round(gamma, 6)
'''
    new='''    # Chapter 5 GIS protocol: disagreement against BOTH rule and feature evidence.
    # Never reduce this to |p - s| alone: that silently drops the rule channel.
    values = (float(p), float(alpha_mean), float(feature_support))
    if any(not math.isfinite(value) for value in values):
        raise ValueError("GIS route evidence must be finite")
    gamma = max(abs(float(p) - float(alpha_mean)), abs(float(p) - float(feature_support)))
    return round(gamma, 6)
'''
    if "import math" not in text:
        text = replace_once(text, "from dataclasses import dataclass\n", "from dataclasses import dataclass\nimport math\n", "alignment math import")
    return replace_once(text, old, new, "alignment GIS route consistency")


def patch_pyproject(text: str) -> str:
    declaration='fuzzyxai-certify = "fuzzyxai.doctoral_cli:main"'
    if declaration in text:
        return text
    anchor='fuzzyxai = "fuzzyxai.cli:main"\n'
    return replace_once(text, anchor, anchor + declaration + "\n", "pyproject command")


TARGETS = {
    "framework/fuzzyxai/fuzzyxai/runtime.py": patch_runtime,
    "framework/fuzzyxai/fuzzyxai/__init__.py": patch_init,
    "framework/fuzzyxai/fuzzyxai/core/risk_observer.py": patch_risk,
    "framework/fuzzyxai/fuzzyxai/risk/risk_function.py": patch_application_risk,
    "framework/fuzzyxai/fuzzyxai/core/alignment.py": patch_alignment,
    "pyproject.toml": patch_pyproject,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true", help="only inspect existing source anchors")
    args = p.parse_args()
    staged: dict[Path, str] = {}
    for name, transform in TARGETS.items():
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f"missing original checkout file: {path}")
        source = path.read_text(encoding="utf-8")
        target = transform(source)
        if name.endswith('.toml'):
            tomllib.loads(target)
        else:
            ast.parse(target, filename=name)
        if source != target:
            staged[path] = target
    if args.check:
        print(f"Preflight OK; would update {len(staged)} existing source files")
        for path in staged:
            print("  ", path.relative_to(ROOT))
        return
    # Preflight completed with no writing, now apply all validated changes.
    for path, value in staged.items():
        path.write_text(value, encoding="utf-8")
        print("Updated", path.relative_to(ROOT))
    print(f"Done. Files edited: {len(staged)}. Run the focused tests before committing.")


if __name__ == "__main__":
    main()
