"""Apply the two existing-file changes for chapter-5 ECG quality handling.

Run from the root of fuzzy-xai-operator after copying this bundle's paths.
Idempotent; refuses to edit if the upstream function boundary has changed.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FUNCTION = '''def build_medical_ecg_signal_route(adapted: AdaptedInput) -> OperatorRoute:
    from fuzzyxai.ecg_quality import assess_ecg_quality

    values = adapted.values
    quality = assess_ecg_quality(values)
    action = quality["action"]
    diagnostic_id = quality["diagnostic_id"]
    diagnostics = []
    if diagnostic_id:
        diagnostics.append({
            "diagnostic_id": diagnostic_id,
            "diagnostic_type": "signal_quality",
            "source": "signal_quality_features",
            "criticality": "medium" if action == "defer_to_human" else "low",
            "message_ru": "ECG technical signal-quality limitation",
            "recommended_action": action,
        })
    computed = {
        "quality_score": float(values["quality_score"]),
        "missing_fragments": int(values["missing_fragments"]),
        "quality_reasons": quality["quality_reasons"],
        "quality_policy": quality["policy_version"],
        "quality_score_kind": values.get("quality_score_kind", "unverified_input_score"),
        "clinical_diagnosis": False,
        "automatic_clinical_accept_allowed": False,
        "diagnostic_id": diagnostic_id,
        "action": action,
    }
    nodes = [
        _node(
            "signal_quality", "ECG signal quality",
            f"quality={values['quality_score']}; missing={values['missing_fragments']}; action={action}",
            "warning" if diagnostics else "passed",
            "Technical input QC only: not an ECG diagnosis or a stenosis prediction.",
            {**values, "quality_reasons": quality["quality_reasons"], "value_source": "computed"},
            "D_signal_quality",
        ),
    ]
    return _route(adapted.scenario_id, "ECG FuzzyXAI OperatorRoute", values, computed, diagnostics, nodes)


'''

REGISTRY_PATH = ROOT / 'framework/fuzzyxai/fuzzyxai/core/scenario_registry.py'
ADAPTER_PATH = ROOT / 'framework/fuzzyxai/fuzzyxai/adapters/signal_quality.py'


def main() -> None:
    original = REGISTRY_PATH.read_text(encoding='utf-8')
    adapter = ADAPTER_PATH.read_text(encoding='utf-8')
    matcher = re.compile(r'def build_medical_ecg_signal_route\(adapted: AdaptedInput\) -> OperatorRoute:\n.*?(?=def build_gd_anfis_shap_route\()', re.S)
    found = list(matcher.finditer(original))
    if len(found) != 1:
        raise RuntimeError('ECG route function cannot be safely located; upstream changed')
    marker = '    required_fields = ("sampling_rate", "quality_score", "noise_segments", "missing_fragments", "baseline_instability")\n'
    validation = '''\n    def validate(self, payload: dict) -> None:
        super().validate(payload)
        from fuzzyxai.ecg_quality import assess_ecg_quality

        assess_ecg_quality(payload)
'''
    if validation not in adapter and adapter.count(marker) != 1:
        raise RuntimeError('ECG adapter field contract changed; edit manually')
    new_registry = matcher.sub(lambda _m: FUNCTION, original, count=1)
    new_adapter = adapter if validation in adapter else adapter.replace(marker, marker + validation, 1)
    if original != new_registry:
        REGISTRY_PATH.write_text(new_registry, encoding='utf-8')
        print('Updated ECG route to dynamic technical quality policy')
    else:
        print('ECG route already updated')
    if adapter != new_adapter:
        ADAPTER_PATH.write_text(new_adapter, encoding='utf-8')
        print('Enabled signal quality payload validation')
    else:
        print('ECG adapter already updated')


if __name__ == '__main__':
    main()
