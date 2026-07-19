# FuzzyXAI Framework Release Status

Status: `v1.2.0rc3` is an untagged Empirical Validation Gate candidate. The measured computational pipeline passes locally. The independent comprehension pilot is `planned_not_run`, and the regulated-domain dictionary is awaiting external semantic review; therefore the release gate is `BLOCKED` and no tag is allowed.

## Measured Computational Gate

- dataset: Breast Cancer Wisconsin (Diagnostic), 569 objects, 30 features, CC BY 4.0;
- split: train 341, validation 114, test 114, seed 42;
- checkpoint model: 30 unique measured SGD states;
- selected case: `case_real_001`, chosen automatically from validation after training;
- forgetting event: epoch 9;
- rare subgroup: smallest of three train-only KMeans clusters, fixed before training/case selection;
- measured native tree rule: `tree_leaf_11`;
- target prediction after leaf suppression: `1 -> 0`;
- test accuracy: `0.947368 -> 0.903509`;
- validation subgroup recall: `0.923077 -> 0.615385`;
- cross-model contracts: logistic regression, decision tree, random forest, fitted Sugeno rules, callable black box;
- black box native rules: 0; tree and Sugeno native rules: present;
- similar-case evidence: one support and one counterexample;
- intervention mode: `sensitivity_analysis`, not an actionable recommendation.

## Controlled Boundary

- `object_85_controlled_story_fixture`: controlled contract and visualization fixture;
- `case_real_001`: measured checkpoint experiment;
- controlled and measured results have different run IDs, directories, and `result_origin` values;
- the research-only image fixture remains controlled and is not medical validation.

## Local Validation

- full Python 3.14 regression: `315 passed`, 409 third-party warnings;
- empirical-focused tests: `16 passed`;
- release-focused tests: `17 passed`;
- Ruff: PASS;
- strict MyPy: PASS;
- operator manifest: `30/30`, PASS;
- deterministic empirical rebuild: PASS;
- Chapter 4 empirical builder/verifier: PASS;
- Chapter 4 measured figures: `3/3` visually inspected;
- public feature CI: pending push;
- public main CI: not run for this candidate.

## Open External Gates

1. A/B comprehension pilot with at least six independent participants.
2. External subject-matter review of the regulated-domain dictionary.

No demonstrated-comprehensibility, clinical-validity, or release-readiness claim is made while either gate is open.

## Archive Policy

- source archive: clean allowlist from committed Git index;
- doctoral archive: separate full historical Git-index export;
- archives must be built only after the release documentation commit;
- no `v1.2.0rc3` tag before all external gates and green feature/main CI.
