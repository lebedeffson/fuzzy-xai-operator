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


## Reproducible developmental comparison (four independent fitted pipelines)

Protocol (fixed in a separate earlier commit):
experiments/ml_lifecycle_development_protocol_v1.json.
Runner: python -m experiments.ml_lifecycle_observer_benchmark.
The experiment deliberately trains and corrupts models **outside** the
observer. Model fit IDs and preprocessing fit IDs are recorded at the
actual sklearn fit invocation. The observer itself does not fit or predict.

Four datasets/model families: Breast Cancer (logistic + scaler), Wine
(random forest classifier), Diabetes (random forest regressor) and Iris
(KMeans). Seed 6072, 75/25 train/evaluation split. A total of **33
pipeline/scenario observations** were produced (some scenarios do not
apply to every task). These observations are correlated within a dataset,
not 33 independently sampled real-world failures.

At code commit 605404d9b5e2d67c0081f5c231e1a178166251bf,
Actions run 43240790172 completed with **26 pytest tests and Ruff passed**
on both Python 3.11 and 3.12. The archived JSON artifacts were separately
downloaded and inspected: the protocol hash, per-scenario outcomes and
ordinary sklearn metrics agree across both Python environments. This is a
scoped CI receipt, not a claim that every unrelated repository job passes.

| Scenario | Cases | Numeric-only baseline alerts | Numeric+ID/schema baseline alerts | Observer alerts |
| --- | ---: | ---: | ---: | ---: |
| Clean data split | 4 | 0 | 0 | 1 warning (Iris identical numeric rows) |
| Model trained on evaluation labels | 3 | 0 | 3 | 3 problems |
| Preprocessor fitted on evaluation inputs | 1 | 0 | 1 | 1 problem |
| Train/evaluation row-ID overlap | 4 | 0 | 4 | 4 problems |
| Reordered evaluation feature columns | 4 | 0 | 4 | 4 problems |
| Copied evaluation row with new ID | 4 | 0 | 0 | 4 warnings |
| Deliberately shifted feature means | 4 | 0 | 0 | 4 warnings |
| Invalid class-probability rows | 2 | 2 | 2 | 2 problems |
| Missing provenance IDs | 4 | 0 | 0 | 4 unknown statuses; Iris also has the inherited duplicate-row warning |
| Permuted ground-truth labels | 3 | 0 | 0 | 0 semantic-fault detections |

The numeric baseline uses sklearn array validation and conventional
classification/regression metrics; the stronger baseline additionally
compares ID and feature-schema declarations. Thus the observer is **not
uniquely better** on straightforward fit provenance and schema cases.
The extra signals are checks of identical numeric input content and
descriptive mean shifts; both are common data-auditing operations, not
a claim of a new universally superior XAI algorithm.

Let the supplied training and evaluation row-ID sets be I_train and I_eval.
The split-overlap statistic is |I_train intersect I_eval|. For fit metadata
it is |I_fit intersect I_eval|. These tests only compare declared IDs.

The exact numeric-row overlap count is
C_dup = sum over evaluation rows of 1[x_eval belongs to X_train].
It is an investigation signal because genuinely repeated observations
may legitimately occur in two partitions (as seen in the Iris split).
The exploratory mean-shift statistic is the maximum over features of
|mean_train - mean_eval| / std_train, with a separate constant-feature
fallback; threshold 2 train SD is a heuristic, not a hypothesis test.

The negative control is particularly important: permuting the *true
labels* can change metrics while keeping all declared shapes and identities
coherent. An observer without an independent trusted source of labels
cannot establish that the label mapping is wrong. It must never claim
universal label-error detection.

The experiment is developer-designed and not blinded or externally
preregistered. It reports controlled detection and important limitations,
not calibrated sensitivity to unknown real-world faults, causal validity,
or model safety. A useful next study would require independent data,
external fit/source provenance, and a predeclared set of unseen failures.
