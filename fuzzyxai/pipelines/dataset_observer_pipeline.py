from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from fuzzyxai import ExplainPlan
from fuzzyxai.data import DatasetRecord, guess_target_column, infer_dataset_profile, split_features_target
from fuzzyxai.risk import RiskAwareModel, RiskPolicy


@dataclass(frozen=True)
class DatasetObserverResult:
    dataset_record: DatasetRecord
    target_column: str
    dataset_profile: Any
    model_name: str
    accuracy: float
    roc_auc: float | None
    case_index: int
    case_prediction: dict[str, Any]
    observer_result: dict[str, Any]
    trace: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if hasattr(self.dataset_profile, 'as_dict'):
            data['dataset_profile'] = self.dataset_profile.as_dict()
        data['dataset_record'] = self.dataset_record.as_trace()
        return data


class DatasetObserverPipeline:
    def __init__(self, model_name: str = 'random_forest', mode: str = 'audit', random_state: int = 42) -> None:
        self.model_name = model_name
        self.mode = mode
        self.random_state = random_state

    def _build_model(self):
        if self.model_name == 'logistic_regression':
            return LogisticRegression(max_iter=2000, random_state=self.random_state)
        if self.model_name == 'random_forest':
            return RandomForestClassifier(n_estimators=120, max_depth=6, random_state=self.random_state)
        raise ValueError(f'Unknown model_name: {self.model_name}')

    def run(self, record: DatasetRecord, df: pd.DataFrame, *, target_column: str | None = None, case_index: int = 0) -> DatasetObserverResult:
        target_column = target_column or record.target_column or guess_target_column(df)
        if target_column is None:
            raise ValueError('Target column is not specified and cannot be guessed. Pass target_column explicitly.')

        profile = infer_dataset_profile(df, requires_audit=(self.mode == 'audit'))
        x_raw, y_raw = split_features_target(df, target_column)
        x_model = pd.get_dummies(x_raw, dummy_na=True)
        y_encoder = LabelEncoder()
        y = pd.Series(y_encoder.fit_transform(y_raw), index=y_raw.index, name=target_column)
        if y.nunique() < 2:
            raise ValueError('Target column must contain at least two classes')

        stratify = y if y.value_counts().min() >= 2 and y.nunique() <= 20 else None
        x_train, x_test, y_train, y_test = train_test_split(
            x_model,
            y,
            test_size=0.25,
            random_state=self.random_state,
            stratify=stratify,
        )
        model = self._build_model()
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        proba = model.predict_proba(x_test) if hasattr(model, 'predict_proba') else None
        accuracy = float(accuracy_score(y_test, pred))
        roc_auc = None
        if proba is not None and proba.shape[1] == 2:
            roc_auc = float(roc_auc_score(y_test, proba[:, 1]))

        plan = ExplainPlan.from_data(x_train, y_train, mode=self.mode).with_reduction_weight(0.10)
        positive_class = 1 if len(y_encoder.classes_) > 1 else 0
        observer = RiskAwareModel(
            model,
            plan=plan,
            policy=RiskPolicy(theta_mid=0.34, theta_high=0.62),
            positive_class=positive_class,
        )
        case_index = max(0, min(int(case_index), len(x_test) - 1))
        case = x_test.iloc[[case_index]]
        observer_row = observer.predict_with_risk(case, metadata={'source': record.source})[0]
        case_prediction = {
            'raw_prediction_encoded': int(pred[case_index]) if hasattr(pred[case_index], 'item') else pred[case_index],
            'raw_prediction_label': str(y_encoder.inverse_transform([int(pred[case_index])])[0]),
            'raw_proba': proba[case_index].tolist() if proba is not None else None,
        }
        return DatasetObserverResult(
            dataset_record=record,
            target_column=target_column,
            dataset_profile=profile,
            model_name=self.model_name,
            accuracy=accuracy,
            roc_auc=roc_auc,
            case_index=case_index,
            case_prediction=case_prediction,
            observer_result=_json_safe_observer_row(observer_row),
            trace={
                'dataset': record.as_trace(),
                'profile': profile.as_dict(),
                'model': self.model_name,
                'mode': self.mode,
                'target_classes': [str(c) for c in y_encoder.classes_],
                'route': 'dataset -> profile -> model -> E_M_ext -> I_pre -> rho -> action',
            },
        )


def _json_safe_observer_row(row: dict[str, Any]) -> dict[str, Any]:
    skip = {'explanation_object', 'risk_decision'}
    safe: dict[str, Any] = {}
    for key, value in row.items():
        if key in skip:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        elif isinstance(value, list):
            safe[key] = [v.item() if hasattr(v, 'item') else v for v in value]
        else:
            safe[key] = str(value)
    return safe


def write_dataset_observer_report(result: DatasetObserverResult, out_dir: str | Path = 'reports/dataset_observer') -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = result.as_dict()
    json_path = out / 'dataset_observer_report.json'
    md_path = out / 'dataset_observer_report.md'
    html_path = out / 'dataset_observer_report.html'
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    md_path.write_text(render_dataset_observer_markdown(data), encoding='utf-8')
    html_path.write_text(render_dataset_observer_html(data), encoding='utf-8')
    return {'json': str(json_path), 'markdown': str(md_path), 'html': str(html_path)}


def render_dataset_observer_markdown(data: dict[str, Any]) -> str:
    profile = data['dataset_profile']
    obs = data['observer_result']
    lines = [
        '# Dataset observer report',
        '',
        f"Dataset: **{data['dataset_record']['dataset_name']}**",
        f"Source: `{data['dataset_record']['source']}`",
        f"Target: `{data['target_column']}`",
        f"Rows: `{profile['n_rows']}`; columns: `{profile['n_columns']}`",
        f"Numeric columns: `{len(profile['numeric_columns'])}`; categorical columns: `{len(profile['categorical_columns'])}`",
        f"Missing rate: `{profile['missing_rate']:.6f}`",
        f"Suggested uncertainty types: `{', '.join(profile['suggested_uncertainty_types'])}`",
        '',
        '## Model',
        '',
        f"- model: `{data['model_name']}`",
        f"- accuracy: `{data['accuracy']:.6f}`",
        f"- roc_auc: `{data['roc_auc']}`",
        '',
        '## Observer result',
        '',
        f"- predicted_risk: `{obs['predicted_risk']:.6f}`",
        f"- uncertainty: `{obs['uncertainty']:.6f}`",
        f"- I_pre: `{obs['pre_interpretability']:.6f}`",
        f"- rho: `{obs['application_risk']:.6f}`",
        f"- action: `{obs['action']}`",
        f"- reason: {obs['reason']}",
        '',
    ]
    return '\n'.join(lines)


def render_dataset_observer_html(data: dict[str, Any]) -> str:
    profile = data['dataset_profile']
    obs = data['observer_result']
    esc = html.escape
    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Dataset observer</title>
<style>body{{font-family:Arial,sans-serif;max-width:980px;margin:32px auto;color:#172033;line-height:1.45}}.card{{border:1px solid #d9e2ec;border-radius:14px;padding:18px;margin:16px 0}}.kpi{{display:inline-block;background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:12px;margin:6px;min-width:145px}}.kpi b{{display:block;font-size:22px;color:#0f766e}}</style></head><body>
<h1>Dataset Observer</h1>
<div class="card"><h2>Dataset</h2><p><b>{esc(data['dataset_record']['dataset_name'])}</b> from <code>{esc(data['dataset_record']['source'])}</code></p><p>rows={profile['n_rows']}, columns={profile['n_columns']}, target=<code>{esc(data['target_column'])}</code></p><p>uncertainty: <code>{esc(', '.join(profile['suggested_uncertainty_types']))}</code></p></div>
<div class="card"><h2>Model</h2><div class="kpi">accuracy<b>{data['accuracy']:.4f}</b></div><div class="kpi">roc_auc<b>{data['roc_auc'] if data['roc_auc'] is not None else 'n/a'}</b></div></div>
<div class="card"><h2>Observer</h2><div class="kpi">risk<b>{obs['predicted_risk']:.4f}</b></div><div class="kpi">u_M<b>{obs['uncertainty']:.4f}</b></div><div class="kpi">I_pre<b>{obs['pre_interpretability']:.4f}</b></div><div class="kpi">rho<b>{obs['application_risk']:.4f}</b></div><div class="kpi">action<b>{esc(obs['action'])}</b></div><p>{esc(obs['reason'])}</p></div>
</body></html>"""
