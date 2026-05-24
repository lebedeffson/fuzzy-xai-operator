from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from fuzzyxai.category import ExplanationCategory, RiskContext, auto_accept_subpresheaf
from fuzzyxai.core.explanation_object import ExplanationObject, Rule, Trace
from fuzzyxai.hierarchy.f0 import F0
from fuzzyxai.risk import RiskPolicy, compute_application_risk

ACTIONS = ['accept', 'lower_confidence', 'request_more_data', 'defer_to_human', 'block']
WEIGHT_KEYS = ['predicted_risk', 'uncertainty', 'interpretability_gap', 'reduction_loss', 'diagnostic']


@dataclass(frozen=True)
class Scenario:
    scenario: str
    z: float
    uncertainty_type: str
    representation: str
    predicted_risk: float
    uncertainty: float
    i_pre: float
    delta: float
    rupture: bool
    expected_action: str


SCENARIOS = [
    Scenario('S0', -1.2, 'none', 'F0', 0.12, 0.10, 0.92, 0.02, False, 'accept'),
    Scenario('S1', 0.1, 'model_margin', 'F0', 0.40, 0.52, 0.86, 0.04, False, 'lower_confidence'),
    Scenario('S2', 0.4, 'interval', 'F_int', 0.45, 0.30, 0.74, 0.42, False, 'request_more_data'),
    Scenario('S3', 0.8, 'trace_gap', 'F_H', 0.58, 0.34, 0.42, 0.18, False, 'defer_to_human'),
    Scenario('S4', 0.2, 'rupture', 'F_N_src', 0.33, 0.40, 0.55, 0.30, True, 'block'),
    Scenario('S5', 1.9, 'high_risk', 'F0', 0.88, 0.30, 0.65, 0.10, False, 'defer_to_human'),
    Scenario('S6', 1.4, 'multi_source_conflict', 'FML-audit', 0.62, 0.72, 0.60, 0.38, True, 'block'),
]


def weight_dict(values: tuple[float, ...]) -> dict[str, float]:
    return dict(zip(WEIGHT_KEYS, values))


def simplex_grid(step: float = 0.05, min_weight: float = 0.05):
    units = int(round(1.0 / step))
    min_units = int(round(min_weight / step))
    for parts in itertools.product(range(min_units, units + 1), repeat=len(WEIGHT_KEYS)):
        if sum(parts) == units:
            yield tuple(p / units for p in parts)


def decision_for(row: Scenario | dict[str, Any], weights: dict[str, float], *, diagnostic: bool | None = None, delta: float | None = None) -> tuple[str, float]:
    get = row.get if isinstance(row, dict) else lambda k: getattr(row, k)
    diagnostics = ['Rupture'] if (get('rupture') if diagnostic is None else diagnostic) else []
    reduction_loss = get('delta') if delta is None else delta
    risk = compute_application_risk(
        get('predicted_risk'), get('uncertainty'), get('i_pre'), reduction_loss, diagnostics, weights
    ).rho
    action = RiskPolicy(theta_mid=0.35, theta_high=0.65, risk_weights=weights).choose_from_risk(
        risk, get('uncertainty'), get('predicted_risk'), get('i_pre'), reduction_loss, diagnostics
    ).action.value
    return action, float(risk)


def baseline_action(row: dict[str, Any], mode: str, weights: dict[str, float]) -> str:
    if mode == 'risk_threshold':
        p = row['predicted_risk']
        return 'defer_to_human' if p >= 0.70 else ('lower_confidence' if p >= 0.35 else 'accept')
    if mode == 'fuzzy_operator':
        i = row['i_pre']
        return 'accept' if i >= 0.60 else ('lower_confidence' if i >= 0.45 else 'defer_to_human')
    if mode == 'observer_no_topos':
        return decision_for(row, weights, diagnostic=False, delta=0.0)[0]
    if mode == 'full_fuzzyxai':
        return decision_for(row, weights)[0]
    raise ValueError(mode)


def calibrate_weights() -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for vals in simplex_grid(0.05):
        weights = weight_dict(vals)
        rows = []
        for sc in SCENARIOS:
            action, rho = decision_for(sc, weights)
            rows.append((action == sc.expected_action, (rho - target_rho(sc.expected_action)) ** 2))
        acc = sum(ok for ok, _ in rows) / len(rows)
        mse = sum(m for _, m in rows) / len(rows)
        distance_to_default = sum((weights[k] - {'predicted_risk': 0.30, 'uncertainty': 0.25, 'interpretability_gap': 0.20, 'reduction_loss': 0.15, 'diagnostic': 0.10}[k]) ** 2 for k in WEIGHT_KEYS)
        score = (acc, -mse, -distance_to_default)
        if best is None or score > best['score']:
            best = {'weights': weights, 'accuracy': acc, 'mse': mse, 'score': score}
    assert best is not None
    best.pop('score')
    return best


def target_rho(action: str) -> float:
    return {'accept': 0.15, 'lower_confidence': 0.40, 'request_more_data': 0.50, 'defer_to_human': 0.75, 'block': 1.0}[action]


def scenario_table(weights: dict[str, float]) -> pd.DataFrame:
    rows = []
    for sc in SCENARIOS:
        action, rho = decision_for(sc, weights)
        rows.append({**asdict(sc), 'rho': rho, 'actual_action': action, 'match': action == sc.expected_action})
    return pd.DataFrame(rows)


def sample_objects(n_per_scenario: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for sc in SCENARIOS:
        for _ in range(n_per_scenario):
            r = asdict(sc)
            r['predicted_risk'] = float(np.clip(r['predicted_risk'] + rng.normal(0, 0.045), 0, 1))
            r['uncertainty'] = float(np.clip(r['uncertainty'] + rng.normal(0, 0.035), 0, 1))
            r['i_pre'] = float(np.clip(r['i_pre'] + rng.normal(0, 0.04), 0, 1))
            r['delta'] = float(np.clip(r['delta'] + rng.normal(0, 0.03), 0, 1))
            rows.append(r)
    return pd.DataFrame(rows)


def baseline_comparison(weights: dict[str, float], n_per_scenario: int, seed: int) -> pd.DataFrame:
    data = sample_objects(n_per_scenario, seed)
    modes = ['risk_threshold', 'fuzzy_operator', 'observer_no_topos', 'full_fuzzyxai']
    rows = []
    for mode in modes:
        pred = [baseline_action(row, mode, weights) for row in data.to_dict('records')]
        exp = data['expected_action'].tolist()
        false_blocks = sum(p == 'block' and e in {'accept', 'lower_confidence'} for p, e in zip(pred, exp))
        missed_ruptures = sum(p in {'accept', 'lower_confidence'} and e == 'block' for p, e in zip(pred, exp))
        rows.append({
            'mode': mode,
            'accuracy': float(np.mean([p == e for p, e in zip(pred, exp)])),
            'false_blocks': int(false_blocks),
            'missed_ruptures': int(missed_ruptures),
            'n': int(len(data)),
        })
    return pd.DataFrame(rows)


def sensitivity(weights: dict[str, float], n_per_scenario: int, seed: int) -> dict[str, pd.DataFrame]:
    data = sample_objects(n_per_scenario, seed + 7)
    out: dict[str, pd.DataFrame] = {}
    wr_rows = []
    for wr in [0.10, 0.15, 0.20, 0.25, 0.30]:
        w = dict(weights)
        rest = 1.0 - wr
        old_rest = sum(v for k, v in w.items() if k != 'diagnostic') or 1.0
        for k in w:
            w[k] = wr if k == 'diagnostic' else w[k] * rest / old_rest
        pred = [baseline_action(row, 'full_fuzzyxai', w) for row in data.to_dict('records')]
        wr_rows.append({'w_R': wr, 'accuracy': float(np.mean(data['expected_action'] == pred)), 'block_rate': float(np.mean(np.array(pred) == 'block'))})
    out['w_R'] = pd.DataFrame(wr_rows)

    theta_rows = []
    for theta_high in [0.60, 0.70, 0.80, 0.90]:
        pred = []
        for row in data.to_dict('records'):
            diagnostics = ['Rupture'] if row['rupture'] else []
            rho = compute_application_risk(row['predicted_risk'], row['uncertainty'], row['i_pre'], row['delta'], diagnostics, weights).rho
            pred.append(RiskPolicy(theta_mid=0.35, theta_high=theta_high, risk_weights=weights).choose_from_risk(
                rho, row['uncertainty'], row['predicted_risk'], row['i_pre'], row['delta'], diagnostics
            ).action.value)
        theta_rows.append({'theta_high': theta_high, 'accuracy': float(np.mean(data['expected_action'] == pred)), 'block_rate': float(np.mean(np.array(pred) == 'block'))})
    out['theta_high'] = pd.DataFrame(theta_rows)

    noise_rows = []
    base_pred = [baseline_action(row, 'full_fuzzyxai', weights) for row in data.to_dict('records')]
    rng = np.random.default_rng(seed + 11)
    for amp in [0.0, 0.02, 0.05, 0.10]:
        noisy = data.copy()
        noisy['i_pre'] = np.clip(noisy['i_pre'] + rng.uniform(-amp, amp, len(noisy)), 0, 1)
        pred = [baseline_action(row, 'full_fuzzyxai', weights) for row in noisy.to_dict('records')]
        noise_rows.append({
            'noise_amp': amp,
            'mean_abs_delta_I': float(np.mean(np.abs(noisy['i_pre'] - data['i_pre']))),
            'action_change_rate': float(np.mean(np.array(pred) != np.array(base_pred))),
        })
    out['noise'] = pd.DataFrame(noise_rows)
    return out


def timing_table(weights: dict[str, float], n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [{'predicted_risk': float(rng.random()), 'uncertainty': float(rng.random()), 'i_pre': float(rng.random()), 'delta': float(rng.random() * 0.4), 'rupture': False} for _ in range(n)]
    configs = [
        ('predict_only', lambda r: r['predicted_risk'] > 0.5),
        ('operator_F0', lambda r: decision_for(r, weights, delta=0.0)[0]),
        ('select_F_int', lambda r: decision_for(r, weights, delta=min(1.0, r['delta'] + 0.05))[0]),
        ('select_F_H', lambda r: decision_for(r, weights, delta=min(1.0, r['delta'] + 0.08))[0]),
        ('select_F_N_src', lambda r: decision_for({**r, 'rupture': r['predicted_risk'] > 0.92}, weights)[0]),
        ('FML_audit', lambda r: decision_for({**r, 'rupture': r['uncertainty'] > 0.90}, weights, delta=min(1.0, r['delta'] + 0.12))[0]),
    ]
    out = []
    for name, fn in configs:
        t0 = time.perf_counter()
        for r in rows:
            fn(r)
        dt = time.perf_counter() - t0
        out.append({'config': name, 'n': n, 'mean_ms': 1000.0 * dt / n, 'total_s': dt})
    return pd.DataFrame(out)


def breast_cancer_validation(weights: dict[str, float], seed: int) -> pd.DataFrame:
    data = load_breast_cancer(as_frame=True)
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25, random_state=seed, stratify=data.target)
    model = RandomForestClassifier(n_estimators=120, max_depth=6, random_state=seed)
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_test)
    pred = model.predict(x_test)
    p_risk = 1.0 - proba[:, 1]  # sklearn target 0 = malignant, so risk = malignant probability
    uncertainty = 1.0 - np.abs(proba[:, 1] - proba[:, 0])
    rows = []
    for p, u in zip(p_risk[:250], uncertainty[:250]):
        rupture = bool(0.45 < p < 0.65 and u > 0.35)
        row = {'predicted_risk': float(p), 'uncertainty': float(u), 'i_pre': float(1 - 0.4 * u), 'delta': float(0.05 + 0.25 * u), 'rupture': rupture}
        expert = 'block' if rupture else ('defer_to_human' if p >= 0.75 else ('lower_confidence' if p >= 0.35 or u >= 0.45 else 'accept'))
        rows.append({'expected_action': expert, 'full_fuzzyxai': baseline_action(row, 'full_fuzzyxai', weights), **row})
    return pd.DataFrame([{
        'dataset': 'sklearn_breast_cancer',
        'n_test': int(len(y_test)),
        'model_accuracy': float(accuracy_score(y_test, pred)),
        'model_roc_auc': float(roc_auc_score(y_test, proba[:, 1])),
        'observer_action_accuracy': float(np.mean([r['expected_action'] == r['full_fuzzyxai'] for r in rows])),
        'simulated_expert_rule': 'risk>=0.75 defer; risk>=0.35 or uncertainty>=0.45 lower; ambiguous high-uncertainty cases block',
    }])


def context_logic_table() -> pd.DataFrame:
    rule = Rule('r_ctx', {'risk': 'medium'}, 'review')
    e = ExplanationObject({'low', 'medium', 'high'}, F0(lambda _x: 0.5, 'risk'), [rule], {rule.name: 0.5}, 0.3, Trace('S4', 'v1', 't0', checksum='S4'))
    cat = ExplanationCategory(gamma_max=0.5)
    e_model = cat.object('E_model', e)
    e_risk = cat.object('E_risk', e)
    morphism = cat.make_morphism(e_model, e_risk, name='T_model_risk', gamma=0.18)
    risk_context = RiskContext(cat, {e_model: {'accept', 'lower_confidence', 'block'}, e_risk: {'request_more_data', 'block'}})
    risk_context.set_restriction(morphism, lambda action: 'block' if action == 'block' else 'lower_confidence')
    auto = auto_accept_subpresheaf(risk_context)
    return pd.DataFrame([
        {'object': e_model.key, 'RiskContext': sorted(risk_context(e_model)), 'AutoAccept': sorted(auto(e_model)), 'restricted_from_E_risk_block': risk_context.restrict(morphism, 'block')},
        {'object': e_risk.key, 'RiskContext': sorted(risk_context(e_risk)), 'AutoAccept': sorted(auto(e_risk)), 'restricted_from_E_risk_block': '-'},
    ])


def _markdown_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ['| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join(['---'] * len(cols)) + ' |']
    for row in df.astype(str).itertuples(index=False):
        lines.append('| ' + ' | '.join(str(v).replace('\n', ' ') for v in row) + ' |')
    return '\n'.join(lines)


def write_outputs(out_dir: Path, tables: dict[str, pd.DataFrame], calibration: dict[str, Any], sensitivity_tables: dict[str, pd.DataFrame]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in {**tables, **{f'sensitivity_{k}': v for k, v in sensitivity_tables.items()}}.items():
        df.to_csv(out_dir / f'{name}.csv', index=False)
    report = {'calibration': calibration, 'tables': {name: df.to_dict(orient='records') for name, df in tables.items()}, 'sensitivity': {name: df.to_dict(orient='records') for name, df in sensitivity_tables.items()}}
    (out_dir / 'chapter5_experiments.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sensitivity_tables['w_R']['w_R'], y=sensitivity_tables['w_R']['accuracy'], mode='lines+markers', name='accuracy(w_R)'))
    fig.add_trace(go.Scatter(x=sensitivity_tables['w_R']['w_R'], y=sensitivity_tables['w_R']['block_rate'], mode='lines+markers', name='block_rate(w_R)'))
    fig.update_layout(title='Chapter 5 sensitivity to rupture weight', xaxis_title='w_R', yaxis_title='rate')
    fig.write_html(out_dir / 'sensitivity_w_R.html', include_plotlyjs='cdn')

    md = ['# Chapter 5 experiments', '', f"Calibration accuracy: **{calibration['accuracy']:.3f}**", '', '## Calibrated weights', '']
    md += [f"- `{k}`: `{v:.3f}`" for k, v in calibration['weights'].items()]
    for title, df in tables.items():
        md += ['', f'## {title}', '', _markdown_table(df)]
    for title, df in sensitivity_tables.items():
        md += ['', f'## sensitivity_{title}', '', _markdown_table(df)]
    (out_dir / 'chapter5_experiments.md').write_text('\n'.join(md), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-per-scenario', type=int, default=1000)
    parser.add_argument('--timing-n', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-dir', default='reports/chapter5')
    args = parser.parse_args()

    calibration = calibrate_weights()
    weights = calibration['weights']
    tables = {
        'scenarios_s0_s6': scenario_table(weights),
        'baseline_comparison': baseline_comparison(weights, args.n_per_scenario, args.seed),
        'timing_complexity': timing_table(weights, args.timing_n, args.seed),
        'breast_cancer_validation': breast_cancer_validation(weights, args.seed),
        'context_logic': context_logic_table(),
    }
    sens = sensitivity(weights, args.n_per_scenario, args.seed)
    write_outputs(Path(args.out_dir), tables, calibration, sens)
    print(json.dumps({'status': 'PASS', 'out_dir': args.out_dir, 'weights': weights}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
