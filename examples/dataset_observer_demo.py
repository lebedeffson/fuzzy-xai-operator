from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

from fuzzyxai.data import CITRegistryDatasetClient, DatasetRecord, guess_target_column, load_table_dataset
from fuzzyxai.pipelines import DatasetObserverPipeline, write_dataset_observer_report


def _load_sample(name: str) -> tuple[DatasetRecord, pd.DataFrame]:
    if name != 'breast_cancer':
        raise ValueError(f'Unknown sample: {name}')
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    # In sklearn target=0 is malignant. We expose a direct binary risk target.
    df['risk_target'] = (df['target'] == 0).astype(int)
    df = df.drop(columns=['target'])
    record = DatasetRecord(
        name='sklearn_breast_cancer',
        source='sklearn.datasets',
        target_column='risk_target',
        task_type='binary_classification',
        description='Breast cancer diagnostic tabular sample; risk_target=1 means malignant.',
        metadata={'sample': name},
    )
    return record, df


def _augment_uncertainty_metadata(
    df: pd.DataFrame,
    *,
    target_column: str | None,
    simulate_intervals: bool,
    simulate_experts: bool,
    simulate_conflict: bool,
) -> pd.DataFrame:
    if not (simulate_intervals or simulate_experts or simulate_conflict):
        return df
    result = df.copy()
    target_column = target_column or guess_target_column(result)
    numeric = [c for c in result.select_dtypes(include='number').columns if c != target_column]
    base = numeric[0] if numeric else None

    if simulate_intervals and base is not None:
        width = float(result[base].std() or 1.0) * 0.05
        result[f'{base}_min'] = result[base] - width
        result[f'{base}_max'] = result[base] + width

    if target_column and target_column in result:
        y = pd.Series(result[target_column]).reset_index(drop=True)
        if simulate_experts:
            result['expert_a'] = y.to_numpy()
            result['expert_b'] = y.mask(y.index % 7 == 0, y.shift(fill_value=y.iloc[0])).to_numpy()
        if simulate_conflict:
            result['source_model'] = y.to_numpy()
            result['source_expert'] = y.mask(y.index % 5 == 0, y.shift(fill_value=y.iloc[0])).to_numpy()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='Run dataset -> model -> XAI observer pipeline')
    parser.add_argument('--sample', default='breast_cancer', help='sample dataset name')
    parser.add_argument('--file', help='local CSV/XLSX/JSON/Parquet file')
    parser.add_argument('--url', help='direct dataset file URL from registry.cit.gov.ru or another source')
    parser.add_argument('--target', help='target column')
    parser.add_argument('--model', choices=['random_forest', 'logistic_regression'], default='random_forest')
    parser.add_argument('--mode', choices=['user', 'audit'], default='user', help='user -> no artificial audit trace; audit -> force trace uncertainty')
    parser.add_argument('--case-index', type=int, default=0)
    parser.add_argument('--out-dir', default='reports/dataset_observer')
    parser.add_argument('--simulate-intervals', action='store_true', help='add *_min/*_max columns to demonstrate FI/FML selection')
    parser.add_argument('--simulate-experts', action='store_true', help='add expert_* columns to demonstrate FH/FML selection')
    parser.add_argument('--simulate-conflict', action='store_true', help='add source_* columns to demonstrate FNsrc/FML selection')
    args = parser.parse_args()

    client = CITRegistryDatasetClient()
    if args.url:
        record = client.download_file(args.url, target_column=args.target)
        df = client.load_dataframe(record)
    elif args.file:
        record = client.from_local_file(args.file, target_column=args.target)
        df = load_table_dataset(Path(args.file))
    else:
        record, df = _load_sample(args.sample)

    target_column = args.target or record.target_column or guess_target_column(df)
    df = _augment_uncertainty_metadata(
        df,
        target_column=target_column,
        simulate_intervals=args.simulate_intervals,
        simulate_experts=args.simulate_experts,
        simulate_conflict=args.simulate_conflict,
    )

    pipeline = DatasetObserverPipeline(model_name=args.model, mode=args.mode)
    result = pipeline.run(record, df, target_column=args.target, case_index=args.case_index)
    paths = write_dataset_observer_report(result, args.out_dir)
    summary = {
        'dataset': result.dataset_record.name,
        'rows': result.dataset_profile.n_rows,
        'features': result.dataset_profile.n_columns - 1,
        'model': result.model_name,
        'accuracy': result.accuracy,
        'roc_auc': result.roc_auc,
        'selected_uncertainty_types': result.dataset_profile.suggested_uncertainty_types,
        'selected_representation': result.observer_result['selected_representation'],
        'representation_class': result.observer_result['representation_class'],
        'application_risk': result.observer_result['application_risk'],
        'safe_action': result.observer_result['action'],
        'I_pre': result.observer_result['pre_interpretability'],
        'I_final': result.observer_result['final_interpretability'],
        'report': paths['html'],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
