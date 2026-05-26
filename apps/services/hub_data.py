from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class HubReports:
    chapter2_summary: dict[str, Any]
    chapter5_summary: dict[str, Any]
    full_pipeline_summary: dict[str, Any]
    category_hott_report: dict[str, Any]
    baseline: pd.DataFrame
    sensitivity_wr: pd.DataFrame
    sensitivity_theta: pd.DataFrame
    timing: pd.DataFrame
    full_predictions: pd.DataFrame


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_reports(root: Path) -> HubReports:
    return HubReports(
        chapter2_summary=read_json(root / 'reports/chapter2/chapter2_breast_cancer_summary.json'),
        chapter5_summary=read_json(root / 'reports/chapter5/chapter5_experiments.json'),
        full_pipeline_summary=read_json(root / 'reports/full_pipeline/summary.json'),
        category_hott_report=read_json(root / 'reports/category_hott/category_hott_checks.json'),
        baseline=read_csv(root / 'reports/chapter5/baseline_comparison.csv'),
        sensitivity_wr=read_csv(root / 'reports/chapter5/sensitivity_w_R.csv'),
        sensitivity_theta=read_csv(root / 'reports/chapter5/sensitivity_theta_high.csv'),
        timing=read_csv(root / 'reports/chapter5/timing_complexity.csv'),
        full_predictions=read_csv(root / 'reports/full_pipeline/predictions.csv'),
    )
