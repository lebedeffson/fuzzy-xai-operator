# FuzzyXAI Framework Release Status

Status: release candidate pending public GitHub Actions.

## Completed

- one installable public framework API;
- 30-entry operator/evidence manifest;
- evidence-first data, training, model-knowledge, decision, counterfactual, and action layers;
- five-panel Matplotlib explanation passport;
- canonical JSON plus MATLAB/Octave loader and dashboard;
- controlled object 85 restoration protocol;
- quarantined generated website;
- clean archive builder based on committed Git state.

## Acceptance gates

See `TEST_REPORT.txt` for the latest local results. A release tag is intentionally withheld until the pushed commit passes the public GitHub Actions matrix.

The clean-checkout acceptance was completed independently on Python 3.11 and
3.12. Both environments reported 299 passing tests, built wheel and sdist
artifacts, and imported the installed wheel outside the source checkout.

## Known limitations

- no native Torch, Keras, or ONNX adapters;
- no claim of support for literally every model;
- MATLAB/Octave consumes the canonical result JSON but does not reimplement the Python operator core;
- domain claims remain limited by the supplied datasets and evidence provenance.
