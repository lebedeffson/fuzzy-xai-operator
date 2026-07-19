# Empirical Validation Gate

## Scope

This milestone separates executable controlled fixtures from measured experiments.

- `object_85_controlled_story_fixture` checks contracts, visualizations, and deterministic wording.
- `case_real_001` is selected automatically from a real checkpoint training run.
- controlled values must not be cited as measured findings.

Every report carries `result_origin`: `measured`, `controlled_fixture`, `derived`, or `expert_defined`.

## Reproduction

```bash
python experiments/real_training_experiment/run_empirical_validation.py
python scripts/build_empirical_chapter4_evidence.py
python scripts/verify_empirical_validation.py
```

The experiment uses the CC BY 4.0 Breast Cancer Wisconsin (Diagnostic) dataset bundled by scikit-learn.
The split and preprocessing are fixed; a three-cluster unsupervised subgroup is defined on train before
checkpoint training and before selection of a forgetting case. The central case is selected from validation,
not test.

## Measured channels

- 30 SGD checkpoints with model fingerprints and partition metrics;
- per-validation-object predictions, confidence, loss, margin, active linear terms, and neighbors;
- automatic correct-to-wrong forgetting events;
- native decision-tree leaf extraction and sibling-fallback suppression;
- train, validation, test, subgroup, critical-error, and calibration effects;
- logistic, tree, forest, fitted Sugeno-rule, and callable black-box contracts;
- one supporting case and one counterexample;
- model sensitivity kept separate from practical actionability.

## Claim boundary

This is a methodological benchmark, not clinical validation. The domain dictionary is versioned and hashed,
but its regulated-domain semantics await independent expert review. The A/B comprehension pilot remains
`planned_not_run`; no comprehensibility claim and no release tag are allowed until at least six independent
participants complete the protocol.
