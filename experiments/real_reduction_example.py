from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from apps.services.layered_case import LayeredCaseService, build_case_state


def _select_case_index(service: LayeredCaseService, target_probability: float) -> int:
    proba = service.backend.model.predict_proba(service.backend.x_test)[:, 0]
    return int(np.argmin(np.abs(proba - float(target_probability))))


def _build_representations(service: LayeredCaseService, idx: int) -> dict[str, Any]:
    x_test = service.backend.x_test
    x_train, _x_val, y_train, _y_val = train_test_split(
        service.backend.x_test,
        service.backend.y_test,
        test_size=0.2,
        random_state=42,
        stratify=service.backend.y_test,
    )

    vec = x_test.iloc[idx].to_numpy(dtype=float)
    p_rf = float(service.backend.model.predict_proba(pd.DataFrame([vec], columns=service.backend.feature_names))[0][0])

    lr = LogisticRegression(max_iter=2000, random_state=42)
    lr.fit(x_train, y_train)
    p_lr = float(lr.predict_proba(pd.DataFrame([vec], columns=service.backend.feature_names))[0][0])

    feature_std = x_train.std(axis=0).replace(0.0, 1.0)
    z = (x_test.iloc[idx] - x_train.mean(axis=0)).abs() / feature_std
    top_features = z.sort_values(ascending=False).head(5).index.tolist()
    std_factor = float(np.clip(np.mean(feature_std[top_features]) / max(1e-6, float(feature_std.mean())), 0.2, 1.5))
    width = float(np.clip(0.06 + 0.04 * std_factor, 0.04, 0.16))

    f0 = {'mu': round(p_rf, 6)}
    f_int = {'low': round(max(0.0, p_rf - width), 6), 'high': round(min(1.0, p_rf + width), 6)}
    f_h = {'values': [round(p_rf, 6), round(p_lr, 6)]}

    support = p_rf
    contradiction = abs(p_rf - p_lr)
    refute = max(0.0, 1.0 - support - contradiction)
    f_n_src = {
        'truth': round(float(np.clip(support, 0.0, 1.0)), 6),
        'indeterminacy': round(float(np.clip(contradiction, 0.0, 1.0)), 6),
        'falsity': round(float(np.clip(refute, 0.0, 1.0)), 6),
    }

    delta_f0 = 0.42
    delta_f_int = abs((f_int['low'] + f_int['high']) / 2.0 - p_rf)
    delta_f_h = abs(np.mean(f_h['values']) - p_rf) + 0.5 * abs(f_h['values'][0] - f_h['values'][1])
    delta_f_n = contradiction * 0.5

    uncertainties = {
        'u_num': True,
        'u_int': width > 0.07,
        'u_expert': abs(p_rf - p_lr) > 0.03,
        'u_conf': contradiction > 0.08,
    }
    needs = [k for k, v in uncertainties.items() if v]
    supports = {
        'F0': {'u_num'},
        'F_int': {'u_num', 'u_int'},
        'F_H': {'u_num', 'u_expert'},
        'F_N_src': {'u_num', 'u_expert', 'u_conf'},
    }
    complexity = {'F0': 1, 'F_int': 2, 'F_H': 2, 'F_N_src': 3}
    deltas = {'F0': delta_f0, 'F_int': delta_f_int, 'F_H': delta_f_h, 'F_N_src': delta_f_n}

    feasible = [c for c in supports if set(needs).issubset(supports[c])]
    if feasible:
        selected_class = sorted(feasible, key=lambda c: (complexity[c], deltas[c]))[0]
    else:
        selected_class = min(supports, key=lambda c: deltas[c])

    return {
        'top_features': top_features,
        'F0': f0,
        'F_int': f_int,
        'F_H': f_h,
        'F_N_src': f_n_src,
        'delta': {
            'F0': round(float(delta_f0), 6),
            'F_int': round(float(delta_f_int), 6),
            'F_H': round(float(delta_f_h), 6),
            'F_N_src': round(float(delta_f_n), 6),
        },
        'selected_class': selected_class,
    }


def generate_real_reduction_example(
    *,
    out_dir: str | Path = 'reports/real_reduction_example',
    target_probability: float = 0.72,
) -> dict[str, Any]:
    service = LayeredCaseService.create()
    idx = _select_case_index(service, target_probability)
    state = build_case_state(service, 'safe', sample_index=idx, dataset_mode='breast_cancer')
    reprs = _build_representations(service, idx)

    chi_auto = bool(state['contexts']['AutoAccept'].get('E_action'))
    result = {
        'object': str(state['input']['sample_id']),
        'P(malignant)': float(state['model']['p_malignant']),
        'selected_features': reprs['top_features'],
        'F0': reprs['F0'],
        'F_int': reprs['F_int'],
        'F_H': reprs['F_H'],
        'F_N_src': reprs['F_N_src'],
        'selected_class': reprs['selected_class'],
        'Delta': reprs['delta'][reprs['selected_class']],
        'Delta_by_class': reprs['delta'],
        'I_pre': float(state['explanation']['I_pre']),
        'rho': float(state['risk']['rho']),
        'action': str(state['risk']['action']),
        'chi_Auto': chi_auto,
        'chi_R': int(state['risk'].get('chi_R', 0)),
        'chi_R_crit': int(state['risk'].get('chi_R_crit', 0)),
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / 'breast_cancer_case.json'
    md_path = out / 'breast_cancer_case.md'
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    md_lines = [
        '# Real reduction example: Breast Cancer',
        '',
        f"- object: `{result['object']}`",
        f"- P(malignant): `{result['P(malignant)']:.6f}`",
        f"- selected_features: `{', '.join(result['selected_features'])}`",
        f"- selected_class: `{result['selected_class']}`",
        f"- Delta: `{result['Delta']}`",
        f"- I_pre: `{result['I_pre']:.6f}`",
        f"- rho: `{result['rho']:.6f}`",
        f"- action: `{result['action']}`",
        f"- chi_Auto: `{result['chi_Auto']}`",
        f"- chi_R: `{result['chi_R']}`",
        f"- chi_R_crit: `{result['chi_R_crit']}`",
    ]
    md_path.write_text('\n'.join(md_lines), encoding='utf-8')
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='reports/real_reduction_example')
    parser.add_argument('--target-probability', type=float, default=0.72)
    args = parser.parse_args()
    result = generate_real_reduction_example(out_dir=args.out_dir, target_probability=args.target_probability)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
