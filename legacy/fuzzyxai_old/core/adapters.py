from typing import Dict, Mapping

def shap_to_linguistic_strength(phi: Mapping[str, float], epsilon: float = 1e-6) -> Dict[str, Dict[str, float]]:
    max_abs = max([abs(v) for v in phi.values()] + [0.0])
    out = {}
    for feature, value in phi.items():
        z = abs(float(value)) / (max_abs + epsilon)
        out[feature] = {
            'weak': max(0.0, 1.0 - z),
            'moderate': max(0.0, 1.0 - abs(2*z - 1.0)),
            'strong': max(0.0, min(1.0, 2*z - 1.0)),
            'sign': 1.0 if value >= 0 else -1.0,
        }
    return out
