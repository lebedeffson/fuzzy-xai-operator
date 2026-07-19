# FuzzyXAI Framework Release Status

Status: `v1.2.0rc1` is a green feature-branch candidate for the Human Explanation Layer milestone. Local framework, deterministic evidence, wheel, and public cross-platform CI gates pass. The external comprehension pilot remains open, so no `v1.2.0-rc1` tag has been created.

## Completed

- typed human cards for decision, reasons, concerns, reliability, action, and tested changes;
- domain dictionary in `ExplainPlan` for features, classes, and actions;
- separate evidence-depth and audience contracts;
- claim grouping, deduplication, comparative language, and domain-user limits;
- support, contradiction, and trust-limitation separation;
- traceability from every visible card to claims and evidence;
- domain-user technical-term exclusion with negative tests;
- object 85 epoch-16 human explanation;
- research-only medical mask-IoU explanation with explicit probability limitation and counterexample;
- four audience profiles plus one-cycle aliases;
- JSON schema and wheel packaging;
- deterministic golden evidence and a `30/30` operator-to-evidence manifest.

## Verified

- local Python 3.14: `308 passed`;
- Ruff: PASS;
- MyPy: PASS;
- `make framework-release-check`: PASS;
- release subset: `26 passed`;
- deterministic evidence build and verifier: PASS;
- isolated wheel import as `1.2.0rc1`: PASS;
- public GitHub Actions run `29691191720`: Python 3.11, Python 3.12, and Octave PASS;
- validated implementation commit: `8b4a865c93b9`.

## Open Gate

The documented comprehension pilot with at least six independent participants remains `planned_not_run`. No demonstrated-comprehensibility claim is made.

## Claim Boundary

- no universal-model-support claim;
- no native Torch, Keras, or ONNX adapter claim;
- no clinical or production-use claim for medical fixtures;
- no interpretation of similarity as diagnosis probability;
- no claim that E0-E5 measures reliability or user comprehension;
- no fabricated statement when evidence or domain language is unavailable.
