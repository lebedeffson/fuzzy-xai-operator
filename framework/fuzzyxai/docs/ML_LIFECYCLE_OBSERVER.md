# FuzzyXAI ML lifecycle observer

The observer only inspects supplied facts about datasets, fitting, and outputs. It never trains, predicts, repairs or changes the source model.

## Evidence categories

- Data: finite tabular features, shape, ordered feature names, class balance, descriptive mean shift.
- Training: declared train/evaluation IDs, observed fit IDs for models and preprocessing, optional loss history, run version and seed.
- Output: prediction shape, classification probability geometry and class mapping, accuracy/macro-F1 when labeled, regression MAE/RMSE/R2, or unsupervised cluster sizes.

## Python API

Import MLRunEvidence and observe_ml_run from fuzzyxai.ml_observer. Construct MLRunEvidence(task='classification', model_family='random_forest', train_features=..., evaluation_features=..., predictions=..., evaluation_targets=...) and call observe_ml_run(evidence).to_dict(). Missing fields remain unknown.

## JSON CLI

python -m fuzzyxai.ml_observer collected_run.json --output observation.json

The JSON object uses the same keys as MLRunEvidence. The caller supplies all evidence, including any row IDs and fit identifiers.

## Interpretation

- A passed individual check does not certify a model.
- Known overlap of source row IDs is reported as a problem. No overlap in declared IDs cannot prove independence of unauthenticated source data.
- Matching feature widths do not establish feature identity: ordered names must be supplied separately.
- Training-history metrics are descriptive, not automatic proofs of convergence or overfitting.
- A shift score is a heuristic, not a statistical drift guarantee.
- Unknown, problem and not_applicable are distinct states.
- This first adapter supports numeric tabular classification, regression and clustering. Images, language models, time series and RAG require other evidence adapters.

The implementation does not alter FuzzyXAI.wrap(...).explain_one(...), existing model behavior or locked scored study artifacts.

Regression: python -m pytest -q tests/test_ml_lifecycle_observer.py
