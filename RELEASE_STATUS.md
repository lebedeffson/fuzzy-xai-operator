# FuzzyXAI Framework Release Status

Status: `v1.1.0rc3` is a feature-branch release candidate for the Explanation Experience milestone. Local framework, cross-model, visualization, packaging, and Chapter 4 evidence gates pass. Public Python 3.11/3.12 and Octave CI pass. The external comprehension pilot remains open, so no `v1.1.0-rc3` tag has been created.

## Completed

- typed evidence, claim, diagnostic, action, and provenance contracts;
- separate claim evidence status, semantic effect, and severity;
- validated `ExplanationGraph` routes from evidence to action;
- E0-E5 capability disclosure with native, surrogate, available, and missing channels;
- typed inspection for claims, rules, concepts, objects, evidence, diagnostics, and actions;
- one typed `ExplanationVisualSpec` shared by ten Matplotlib and ten Plotly views;
- 12-checkpoint object 85 trajectory with a controlled forgetting event;
- measured train, validation, test, subgroup, critical-error, and calibration rule ablation;
- controlled black-box, sklearn linear, tree, ANFIS, and research-only image scenarios;
- deterministic golden evidence and a 30/30 operator-to-evidence matrix;
- Chapter 4 evidence archive with 12 figures and machine-readable tables;
- full committed-tree release packaging with the generated website quarantined.

## Verified Locally

- Python 3.14 working tree: `304 passed`;
- clean committed snapshot on Python 3.12: `304 passed`;
- public-contract Ruff gate: PASS;
- strict MyPy contract gate: PASS;
- `make framework-release-check`: PASS;
- operator manifest: `30/30`, PASS;
- deterministic golden evidence double build: PASS;
- explanation experience verifier: PASS;
- wheel and sdist build: PASS;
- installed wheel import outside checkout: `1.1.0rc3`, PASS.
- public GitHub Actions run `29685440340`: Python 3.11, Python 3.12, and Octave PASS.

## Open Gates

- documented comprehension pilot with at least six external participants.

The pilot is `planned_not_run`. No user-comprehension claim is made. Public run `29685440340` validates implementation commit `db7a174`.

## Claim Boundary

- no universal-model-support claim;
- no native Torch, Keras, or ONNX adapter claim;
- surrogate channels are labeled and carry fidelity limitations;
- similarity identifies representation and metric and is not a diagnosis probability;
- the medical fixture is research-only;
- missing evidence produces `insufficient_evidence` or review, never invented metrics;
- E0-E5 describes evidence depth, not model quality.
