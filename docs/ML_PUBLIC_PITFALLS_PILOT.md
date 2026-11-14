# Pilot: external-source ML preprocessing failures (developmental)

## Research question

Can the **unchanged** FuzzyXAI read-only ML observer identify two externally
documented classes of preprocessing error in *actual fitted sklearn
pipelines*, while abstaining when the necessary transform witness is missing?
Can it outperform a direct rule with the same underlying observed data? We
must explicitly test this second question instead of assuming superiority.

## Provenance and registration

The exact scenarios, model/data choices and decision rules were preregistered
in `experiments/ml_public_pitfalls_protocol_v1.json` in commit
`987179137a3a927872f53a07fb59ffc077fe9844`, **before** the replay code
and outcomes were produced.

External source: [scikit-learn, Common pitfalls and recommended practices,
v1.9.1](https://scikit-learn.org/stable/common_pitfalls.html). In particular:
§12.1 inconsistent preprocessing; §12.2.1 fitting preprocessors exclusively
on training data; §12.2.2 supervision leakage through feature selection.

The examples originate in **independently authored public documentation**;
their code has been adapted by the FuzzyXAI researchers for explicit recorder
instrumentation and fixed protocols. They are **deliberately incorrect
pedagogical examples**, not real-world incidents discovered blindly, not an
external laboratory's evaluation and not proprietary production telemetry.

## Executed protocols

| Externally documented error | Actual computation | Valid control | Missing-evidence control |
| --- | --- | --- | --- |
| Inconsistent inference-time scaling | StandardScaler trained on regression train batch, LinearRegression predicts unscaled test row | Predict transformed evaluation | Same correct execution, transform witness withheld |
| Supervised feature-selection leakage | SelectKBest fitted on features **and labels** of all 200 rows before holdout model fit | Fit selector on training rows only | Same correct execution, transform witness withheld |
| Preprocessing before splitting | StandardScaler statistics fitted on train plus evaluation data in diabetes/Ridge | Fit scaler on training rows only | Same correct execution, transform witness withheld |

Total: three source patterns × three executions = nine correlated,
developer-controlled records. The actual fitted estimator is invoked via
`FitRecorder`, the actual transform via `TransformRecorder`, and inference
via `OutputRecorder`. `observe_ml_run` receives completed evidence only;
it never performs training, inference, repair or scoring autonomously.

## Primary outcome and baselines

The protocol fixes the decisive check for each invalid case **before**
execution. Compare three decisions based on the same case:

- **FuzzyXAI**: problem/unknown from named provenance checks.
- **Equal-evidence direct rules**: exact feature-array and fit-scope
  comparisons with access to the same witnessed model/transform/input data.
  A detected fault here is not an algorithmic FuzzyXAI advantage.
- **Metadata-only**: only train/eval/fit row IDs without content evidence.
  It intentionally has less information and is not a fair superiority
  baseline.

Report TP/FN on documented incorrect paths, FP/TN on correct paths and
**abstention** on missing-transform records. Never treat a missing witness as
a valid certificate. Record model MSE or accuracy separately to show a real
trained model was used, **not** to define whether its provenance is correct.
Do not inflate the effective sample size by counting repeated measurements
from the same documented pattern as independent incidents.

## Boundaries not resolved by this pilot

Real uncontrolled incident reports, untrusted caller identities,
compromised recorder and key custody, cross-run warm-start leakage,
oversampling before split, non-tabular data, multi-stage transformations,
and independent blinded evaluation remain outside this experiment.
The [scikit-learn warm-start leakage report #14034](https://github.com/scikit-learn/scikit-learn/issues/14034)
illustrates an important cross-run failure mode for future work; it is
**not executed or scored by this pilot**.

A green CI means the declared comparison **ran reproducibly**. A negative
or tied comparator outcome is a genuine scientific result, never a
reason to alter the preregistered dataset, thresholds or examples.

## Reproduction

```bash
python -m pip install -e ".[dev]"
python -m pytest -q tests/test_ml_public_pitfalls_replay.py
python -m experiments.ml_public_pitfalls_replay > ml_public_pitfalls_result.json
```

CI runs Python 3.11 and 3.12 and uploads both results with the protocol
and source code for independent audit.

## Measured outcomes (post-registration)

The source protocol was **unchanged** after its registration. Evaluated by
[GitHub Actions 43632310414](https://github.com/lebedeffson/fuzzy-xai-operator/actions/runs/43632310414),
**SUCCESS** on Python 3.11 and 3.12; independently downloaded artifacts
`14127684512` and `14128656924` contain the same exact protocol SHA256
`25c54b610db414c2572ac467403080b7e6cf092a163592dd9a5f75dec456d128`.

| Predeclared decision source | Known errors detected | False alerts / 3 valid | Abstentions / 3 missing |
| --- | ---: | ---: | ---: |
| FuzzyXAI | 3/3 | 0 | 3/3 |
| Equal-evidence direct array rules | 3/3 | 0 | 3/3 |
| Metadata-only ID baseline | 2/3 | 0 | 0/3 (reports `no_alert`, not verified clean) |

The direct comparator ties every observer decision; **no intrinsic detection
advantage** over equally informed rules was demonstrated. The observer
provides an explicit unavailable/unknown state, which the deliberately
weaker metadata-only baseline here lacks.

Actual model-output diagnostics:

| Case | Valid-path metric | Error-path metric | Interpretation |
| --- | ---: | ---: | --- |
| Inference without StandardScaler | MSE 0.90280 | MSE 62.80867 | Real performance degradation, with an independently observable transform-to-inference mismatch |
| Supervised feature selection before split | Accuracy 0.50 | Accuracy 0.76 | Artificially optimistic held-out evaluation, because the feature-selection stage saw evaluation labels |
| Fitted scaler on evaluation rows | MSE 2842.83463 | MSE 2842.89265 | Proven provenance violation despite negligible change in model MSE |

Performance differences are *observations*, **not** the decision criterion
for leakage. The third row matters because predictive metrics would not
reliably reveal this misuse. This is an example-level result, not a
statistical effect-size estimate across independent incidents.

Python 3.11 and 3.12 use different NumPy patch versions. The case labels,
all observer/baseline decisions and aggregate counts are identical. Numerical
model metrics agree at the displayed precision, **not bit-for-bit**:
the diabetes/Ridge clean and withheld-witness runs give
MSE 2842.834633538595 (3.11) versus 2842.834633538591 (3.12),
an absolute difference of about 4e-12. Thus the JSON outputs differ
in both environment metadata and those two floating-point observations.

### What the result does not establish

- No independently selected natural failure incidents or blinded researcher.
- No sensitivity/specificity confidence intervals are justified from only
  three developer-adapted, correlated example families.
- No superiority over rules with equivalent witness access.
- No assurance of correct origin/authenticity when the caller supplying
  records is compromised.
- The dataset and targets are the ones in published educational demonstrations,
  not field data from an external organization.
