# FuzzyXAI Framework Release Status

Status: `v1.0.0-rc1` remains the validated public baseline; `v1.1.0rc1` Explanation Experience is a local implementation candidate pending public CI.

## Completed

- one installable public framework API;
- 30-entry operator/evidence manifest;
- evidence-first data, training, model-knowledge, decision, counterfactual, and action layers;
- claim-centered `ExplanationGraph` and `ExplanationClaim` contract;
- E0-E5 explanation-level and channel disclosure;
- focused Matplotlib/Plotly views driven by `ExplanationVisualSpec`;
- canonical JSON plus MATLAB/Octave loader and dashboard;
- controlled object 85 restoration protocol;
- quarantined generated website;
- clean archive builder based on committed Git state.

## Acceptance gates

See `TEST_REPORT.txt` for the latest local and public results. Public run
`29668215392` and final metadata run `29668320459` passed the Python 3.11,
Python 3.12, and Octave jobs. The validated release tag is `v1.0.0-rc1`.

The `v1.0.0-rc1` clean-checkout acceptance was completed independently on Python 3.11 and
3.12. Both environments reported 299 passing tests, built wheel and sdist
artifacts, and imported the installed wheel outside the source checkout.

The v1.1 candidate adds controlled object-85, ANFIS, and research-only medical golden explanations. Its comprehension study is explicitly `planned_not_run`; no user-understanding claim is made yet. A `v1.1.0-rc1` tag must not be created until the branch is committed, pushed, and public Python/Octave CI is green.

## Known limitations

- no native Torch, Keras, or ONNX adapters;
- no claim of support for literally every model;
- MATLAB/Octave consumes the canonical result JSON but does not reimplement the Python operator core;
- domain claims remain limited by the supplied datasets and evidence provenance.
