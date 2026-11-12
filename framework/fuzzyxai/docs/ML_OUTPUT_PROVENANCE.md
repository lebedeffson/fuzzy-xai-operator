# FuzzyXAI — read-only provenance of model output calls

## Scope

The ML lifecycle observer can check output dimensions and task metrics, but
those do not establish whether a reported array was actually returned by
`predict`/`predict_proba` for the claimed evaluation rows.

`OutputRecorder` is a small opt-in wrapper. **Only the caller** invokes
inference through `recorder.predict(...)` or `recorder.predict_proba(...)`.
These wrappers forward exactly one requested estimator method call, return
the result unchanged, and append evidence only after a successful return.
**The observer itself never calls fit, predict, predict_proba or repair.**

## Integration

```python
from fuzzyxai.ml_observer import MLRunEvidence, observe_ml_run
from fuzzyxai.output_provenance import OutputRecorder

outputs = OutputRecorder()
ids = tuple(f'eval:{i}' for i in evaluation_indices)
predicted = outputs.predict(model, X_test, row_ids=ids)
probabilities = outputs.predict_proba(model, X_test, row_ids=ids)

record = MLRunEvidence(
    task='classification',
    model_family=type(model).__name__,
    evaluation_features=X_test,
    evaluation_targets=y_test,
    evaluation_ids=ids,
    predictions=predicted,
    probabilities=probabilities,
    probability_classes=tuple(str(x) for x in model.classes_),
    output_records=[event.to_dict() for event in outputs.records],
)
report = observe_ml_run(record)
for check in report.checks:
    if check.code.startswith('output_trace'):
        print(check.code, check.status, check.detail)
```

## Observed dimensions

- The ordered reported evaluation IDs must match the observed call IDs.
- For **raw finite numeric features**, the ordered batch of evaluation
  features must match the values supplied at the actual inference call.
- The reported numeric output (1D or 2D) is compared to the actual
  recorded result by a shape-bound SHA256 fingerprint.
- For `predict_proba`, the textual class-column order from `classes_`
  is compared to separately declared `probability_classes`.
- Missing recorded calls, unsupported nonnumeric output types, transformed
  inputs without verified transforms, and unselected multiple calls
  explicitly remain `unknown`.

These checks use **one full evaluation batch**. They do not silently stitch
together partial inference calls or assume batch-order alignment.

## Negative controls and limitations

A valid output witness can accompany a poor or biased classifier. It
confirms that the report is internally consistent with an observed call,
**not** that the model is correct, that its fitted state is authenticated,
or that reference labels are true. No external attestation is provided:
a compromised caller or recorder can manufacture a consistent trace.

Numeric output SHA256 fingerprints are not cryptographic confidentiality
for small, low-entropy prediction arrays. The raw output values are not
serialized in the witness, but users must not treat the digest as
privacy-preserving or distribute sensitive witness bundles unrestricted.

Outputs with string class labels are not yet certified by this numeric
adapter: they return unknown instead of silently coercing type identity.
Arbitrary image, text, LLM/RAG and generative outputs are not supported.

## Development comparison

Protocol: `experiments/ml_output_witness_development_protocol_v1.json`.
Runner: `python -m experiments.ml_output_witness_development_benchmark`.
Tests: `tests/test_ml_output_provenance.py` and
`tests/test_ml_output_witness_benchmark.py`.

The experiment uses actual fitted scikit-learn classification, regression
and clustering models with developer-created output perturbations, including
row swaps and probability-column permutations that preserve structural
properties. A baseline with direct access to actual model-output call
inputs and values can detect the same defects. This is not a blinded study
of real-world incident sensitivity or evidence of general superiority.


## Verified developmental results (12 November 2026)

The protocol was committed before implementing the benchmark at
experiments/ml_output_witness_development_protocol_v1.json.
Its SHA256 is
1520580a0b640c258ee5204244cfe54a388fdea4ce1614caf45f982c82d7e79a.

An initial CI run at commit 74dd16c4 failed only one benchmark assertion:
the plain numeric comparator cast integer class predictions to float and
then compared strings such as "1.0" against canonical class names "1".
This mistakenly flagged two clean classifiers. The baseline was repaired
to compare actual numeric labels with numeric class values; the
prespecified protocol was NOT altered. The test was also corrected to
give the ordinary classifier probability/argmax consistency check credit
for the two real swapped-prediction violations it detects.

GitHub Actions run 43293095587 at tested source commit
0c56781b77387f8e01622a070979ac16d811e5a9 completed successfully:
**Python 3.11: 79 pytest PASS + Ruff PASS;
Python 3.12: 79 pytest PASS + Ruff PASS**.
Both archived JSON artifacts were downloaded and opened:
- ml-output-witness-3.11, artifact ID 13977288753,
  Python 3.11.18, NumPy 2.4.6, scikit-learn 1.9.1;
- ml-output-witness-3.12, artifact ID 13979275683,
  Python 3.12.16, NumPy 2.5.4, scikit-learn 1.9.1.

Protocol SHA256, all 22 case records, per-scenario counts,
comparison baseline outcomes and scientific limits match exactly.
The JSON byte hashes differ only because version/environment data
differ; results are semantically identical.

| Scenario | Cases | Ordinary numeric baseline | Output witness | Equally privileged direct comparator |
| --- | ---: | ---: | ---: | ---: |
| Clean | 4 | 0 | 0 | 0 |
| Swapped predictions | 4 | 2 | 4 | 4 |
| Reordered evaluation feature rows | 4 | 0 | 4 | 4 |
| Reordered evaluation row IDs | 4 | 0 | 4 | 4 |
| Missing output call witness | 4 | 0 | 4 UNKNOWN, no claimed detection | 0 |
| Probability columns swapped together with class names | 2 | 0 | 2 | 2 |

The 22 cases are controlled modifications of only four real sklearn
model/dataset pipelines. They are correlated and developer-constructed,
not independent real-world incidents. The measured advantage relative to
ordinary shape/finite/probability checks comes from additional source
evidence. **An equally privileged direct comparison can flag every
altered output that the witness detects.** No universal model-fault
detection, prediction-accuracy improvement, label-truth verification,
model identity attestation, or generality to LLM/CV/RAG is established.

A complete real caller-initiated path is covered by
tests/test_ml_observer_end_to_end.py: feature data, actual fit and target
witnesses, explicit predict/predict_proba, persisted JSON evidence,
read-only audit and separate injected corruption of targets versus outputs.
All previous observer/fit/target and public API regression tests remain in
the same 79-test matrix.
