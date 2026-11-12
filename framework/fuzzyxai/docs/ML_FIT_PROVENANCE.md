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
