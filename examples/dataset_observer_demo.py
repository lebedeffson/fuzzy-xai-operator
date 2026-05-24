from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

from fuzzyxai.data import CITRegistryDatasetClient, DatasetRecord, load_table_dataset
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


def main() -> None:
    parser = argparse.ArgumentParser(description='Run dataset -> model -> XAI observer pipeline')
    parser.add_argument('--sample', default='breast_cancer', help='sample dataset name')
    parser.add_argument('--file', help='local CSV/XLSX/JSON/Parquet file')
    parser.add_argument('--url', help='direct dataset file URL from registry.cit.gov.ru or another source')
    parser.add_argument('--target', help='target column')
    parser.add_argument('--model', choices=['random_forest', 'logistic_regression'], default='random_forest')
    parser.add_argument('--case-index', type=int, default=0)
    parser.add_argument('--out-dir', default='reports/dataset_observer')
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

    pipeline = DatasetObserverPipeline(model_name=args.model)
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
        'application_risk': result.observer_result['application_risk'],
        'safe_action': result.observer_result['action'],
        'I_pre': result.observer_result['pre_interpretability'],
        'I_final': result.observer_result['final_interpretability'],
        'report': paths['html'],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
