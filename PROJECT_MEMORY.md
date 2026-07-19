# Project Memory

- Date: 2026-07-19
- Branch: `feat/empirical-validation-gate`
- Base release candidate: `1ef84d88733bdc94d1221b6f2bb6992452d0347d`
- Empirical implementation commit: `cb4b433b414efdc2b6fa1346fc7086576dd52c26`
- Candidate version: `1.2.0rc3`
- Release tag: not created

## Current Focus

The primary product remains the installable FuzzyXAI research framework. The website is quarantined. The current milestone validates the evidence pipeline on real checkpoint training while preserving explicit blockers for external human comprehension and regulated-domain semantics.

## Measured Experiment

- BCDW dataset through the scikit-learn bundled loader: 569 objects, 30 features, CC BY 4.0;
- fixed seed-42 split: 341 train, 114 validation, 114 test;
- train-only standardization and a rare subgroup defined before training as the smallest of three train KMeans clusters;
- 30 unique SGD checkpoint fingerprints;
- validation case selected automatically after training as `case_real_001` (source row `bcwd_0215`);
- measured correct-to-wrong forgetting event at epoch 9;
- native tree leaf `tree_leaf_11` suppressed with an explicit sibling-branch fallback;
- target prediction changes `1 -> 0`;
- test accuracy changes `0.947368 -> 0.903509`;
- validation subgroup recall changes `0.923077 -> 0.615385`;
- every number is generated from the run and serialized, not inserted into a golden JSON.

## Empirical Contracts

- typed `TrainingCheckpointEvidence` and `RuleAblationEvidence`;
- typed domain feature, semantic validation, comparison, similar-case, and counterfactual explanation contracts;
- sample-size-aware wording suppresses percentile claims for small references;
- semantic direction conflicts are rejected;
- unreviewed regulated-domain language yields `insufficient_domain_language`;
- similar-case output is limited to one support and one counterexample;
- sensitivity analysis is separate from a domain-validated actionable counterfactual;
- black-box callable receives no native rules;
- tree and fitted Sugeno-rule models expose measured native structures.

## Scenario Boundary

- `object_85_controlled_story_fixture`: controlled contract/visualization fixture, never a measured result;
- `case_real_001`: measured validation case selected by the forgetting algorithm;
- medical image evidence remains a controlled research-only channel test;
- every Chapter 4 result must declare `result_origin`.

## Validation

- full local Python 3.14 regression: `315 passed`;
- empirical focused suite: `16 passed`;
- release focused suite: `17 passed`;
- Ruff and strict MyPy: PASS;
- operator manifest: `30/30`, PASS;
- deterministic empirical manifest after two runs: PASS;
- Chapter 4 empirical package and manifest verifier: PASS;
- Chapter 4 empirical ZIP SHA256: `f80aa4ba799e91b492a10553e7f12c6ebe0e7572a226d6dd562cb8be3973b9e4`;
- measured figures: `3/3`, visually inspected;
- public feature CI: pending push;
- public main CI: not run for this candidate.

## Open Gates

- independent A/B comprehension pilot: `planned_not_run`, zero participants recorded;
- regulated-domain dictionary review: `insufficient_domain_language`, external reviewer required;
- release gate: `BLOCKED`;
- tag: forbidden until pilot, semantic review, feature CI, and main CI pass.

## Archive Policy

- build the clean source release with `python scripts/build_framework_release.py` only from a committed index;
- build the separate historical archive with `python scripts/build_doctoral_research_archive.py`;
- do not package the dirty worktree;
- publish each archive's commit, file count, manifest, and SHA256.

## Next Step

Push the feature branch and verify public Python 3.11/3.12 and Octave CI. Then run the independent pilot and obtain an external semantic review. Do not merge, tag, or finalize Chapter 4 claims before those external gates are recorded.
