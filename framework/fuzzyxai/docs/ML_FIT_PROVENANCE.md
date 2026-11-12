# Observed sklearn fit calls: optional provenance evidence

## Goal

The existing read-only ML lifecycle observer can inspect manually supplied
model_fit_ids and preprocessor_fit_ids. Those declarations may be incorrect.
An opt-in FitRecorder captures the inputs of an *actual caller-invoked*
sklearn fit call, including source IDs, finite numeric row fingerprints,
estimator role, and raw/transformed feature-space type.

No automatic scheduling, retraining, repairs, decisions, or model corrections
are performed. The caller still initiates each estimator.fit call explicitly
through the wrapper, which delegates it once and records only success.

## Python integration

    from fuzzyxai.fit_provenance import FitRecorder
    from fuzzyxai.ml_observer import MLRunEvidence, observe_ml_run
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    recorder = FitRecorder()
    scaler = recorder.fit(
        StandardScaler(), X_train,
        role="preprocessor", row_ids=tuple(train_ids),
        feature_space="raw",
    )
    model = recorder.fit(
        LogisticRegression(max_iter=700),
        scaler.transform(X_train), y_train,
        role="model", row_ids=tuple(train_ids),
        feature_space="transformed",
    )
    evidence = MLRunEvidence(
        task="classification", model_family="logistic_regression",
        train_features=X_train, evaluation_features=X_test,
        train_ids=tuple(train_ids), evaluation_ids=tuple(test_ids),
        preprocessor_applied=True,
        fit_records=tuple(item.to_dict() for item in recorder.records),
    )
    report = observe_ml_run(evidence)

The observer reads these fit records. It never calls fit, predict, or
transform itself. FitRecorder only forwards a fit *explicitly requested*
by caller code; a failed fit creates no successful witness.

## What is checked

- Fit isolation: witnessed model and preprocessing fit IDs are compared
  to evaluation IDs, rather than only trusting the declared fit-ID list.
- Input content: raw numeric rows are hashed and compared by source ID
  against independently supplied train/evaluation feature matrices.
  Wrong source IDs with the right count may then be detected.
- Declaration disagreement: witnessed IDs are compared to optional manual
  fit-ID declarations, returning a warning on mismatch.
- Missing witnesses remain UNKNOWN; no authentication is invented.
- A model fit on transformed features cannot be matched to raw feature
  bytes without a separately observed transformation chain. Content
  remains UNKNOWN, not OK.
- No declared fitted preprocessor is NOT_APPLICABLE only when there was
  no contradictory recorded preprocessing call.

## Source-content representation and limits

Numeric features are canonicalized to finite little-endian float64, with
signed zeros normalized and a version/width domain separator. Per-row SHA256
values and a dimension-bound ordered batch SHA256 are computed before fit,
and only the digests and declared row IDs are retained in the trace.

These hashes detect inconsistencies between two copies of supplied evidence.
They do NOT establish the true origin of a dataset or attest to a trusted
runtime: if someone controls the caller's source IDs, arrays and trace
collector, consistent false records can be manufactured. Uninstrumented
fit calls are invisible. No target labels or raw feature values are stored
by this recorder.

## Developer-designed experiment

    python -m pytest -q tests/test_fit_provenance.py tests/test_ml_fit_source_witness_benchmark.py
    python -m experiments.ml_fit_source_witness_benchmark

Protocol: experiments/ml_fit_source_witness_protocol_v1.json.
Four fitted sklearn pipelines and five deliberate cases each:

- normal training from the declared training partition;
- evaluation rows fit with true IDs;
- evaluation rows fit but falsely assigned training IDs;
- evaluation rows witnessed accurately while manual fit-ID metadata lies;
- legitimate fit without any witness (insufficient evidence).

Comparison baseline: a competent train/evaluation fit-ID overlap checker.
The new check does not claim novelty over a system that already observes
authenticated fit events and source content. Results are correlated
developer-created fault injections, NOT independent blind accuracy
measurements, proofs of model quality, or unknown-fault detection.


## Measured developmental comparison, 12 November 2026

The original protocol file was committed before the benchmark was run:
experiments/ml_fit_source_witness_protocol_v1.json, SHA256
dc7cdfb78d3a2ff3540ba67152042bcc3966ff5f0021d042fa48f7c9a264e93b.
The first benchmark implementation mistakenly used tree/RandomForest
estimators instead of the declared LogisticRegression and LinearRegression.
**Those original results were not accepted as protocol-compliant.**
The benchmark was repaired to instantiate the protocol's exact estimator
types. For unscaled LogisticRegression, solver=newton-cholesky and
max_iter=800 were used after an LBFGS convergence warning was observed.
This is a developmental solver choice, not a preregistered comparison
of optimizers.

Corrected source commit f1fe1528cfc1e2bbf141eb60066bd84364cd1395
passed [GitHub Actions 43278096157](https://github.com/lebedeffson/fuzzy-xai-operator/actions/runs/43278096157)
on Python 3.11 (scikit-learn 1.9.1, NumPy 2.4.6) and Python 3.12
(scikit-learn 1.9.1, NumPy 2.5.4): **37 tests and Ruff passed
in each environment**. Both archived benchmark JSON files were downloaded
and compared; they agree by dataset, scenario, status and protocol hash.
All recorded LogisticRegression fits converged.

| Scenario | Cases | Manual fit-ID baseline alerts | Witness alerts |
|---|---:|---:|---:|
| Only train inputs, honest IDs | 4 | 0 | 0 |
| Evaluation data included, honest fit IDs | 4 | 4 | 4 |
| Evaluation values disguised under train IDs | 4 | 0 | 4 (raw content mismatch) |
| Evaluation fit witnessed, manual report hides its IDs | 4 | 0 | 4 |
| No instrumented fit record | 4 | 0 | 0; 4 unknown |

These are 20 developer-created scenarios on four model/dataset pipelines,
not 20 statistically independent environments. This is a software
fault-injection result, not estimated sensitivity to real-world unknown
leakage. The caller can forge both source IDs and data; hashes are not
external attestations. A witness does not authenticate supervised target
labels. The shuffled-label negative control still passes structural
source-content checks, as expected.

The defensible contribution is **observing explicit caller-initiated
training inputs and detecting contradictions between fit-value content,
declared identity and manual metadata** without modifying the model.
