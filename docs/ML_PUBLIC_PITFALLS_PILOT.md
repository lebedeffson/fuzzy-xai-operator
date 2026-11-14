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
