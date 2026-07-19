# FuzzyXAI Framework Release Status

Status: `v1.2.0rc2` is a green feature-branch candidate for the Human Explanation Quality Gate milestone. Local framework, deterministic evidence, wheel, and public cross-platform CI gates pass. The external comprehension pilot remains open, so no `v1.2.0-rc2` tag has been created.

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
- mandatory subject, effect direction, and comparison for every domain-user reason;
- structured reliability support, limitations, missing evidence, and conclusion;
- complete before/after counterfactual cards; incomplete interventions are hidden from the first level;
- direct-feature-first ranking and a ban on vague first-level phrases;
- explicit `insufficient_domain_language` for technical class codes without domain meaning;
- executable, non-fabricating A/B comprehension-pilot scorer and response template;
- deterministic golden evidence and a `30/30` operator-to-evidence manifest.
- deterministic Chapter 4 archive with `12/12` figures and `4/4` embedded human explanations.

## Verified

- local Python 3.14: `311 passed`;
- Ruff: PASS;
- MyPy: PASS;
- `make framework-release-check`: PASS;
- release subset: `26 passed`;
- deterministic evidence build and verifier: PASS;
- unpacked wheel payload import as `1.2.0rc2` against the validated dependency environment: PASS;
- Chapter 4 evidence ZIP SHA256: `e5cfe3ba290ba7966e74a4dddb28ead16a9a8846eb9d184d518facb2a499b965`;
- public GitHub Actions run `29693479788`: Python 3.11, Python 3.12, and Octave PASS;
- validated implementation commit: `97e25e3990f4`.

## Open Gate

The documented comprehension pilot with at least six independent participants remains `planned_not_run`. No demonstrated-comprehensibility claim is made.

## Claim Boundary

- no universal-model-support claim;
- no native Torch, Keras, or ONNX adapter claim;
- no clinical or production-use claim for medical fixtures;
- no interpretation of similarity as diagnosis probability;
- no claim that E0-E5 measures reliability or user comprehension;
- no fabricated statement when evidence or domain language is unavailable.
