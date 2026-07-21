# Q1 empirical remediation analysis plan

Base commit: `cafe403c7d60e36b08f56a5325ba380718a5be35`.

This plan is frozen before the new held-out test partitions are evaluated. The prior E1-E8 package is an immutable baseline, including its negative rule-ablation and critical-rupture results.

## Partitions and leakage boundary

- Model fitting uses `train` only.
- Feature, rule, threshold, cost-scenario and cascade selection use `train` and `validation` only.
- The final `test` partition is evaluated once after writing the calibration freeze record.
- Every operation records partition identifiers and hashes. A test partition in any `fit` or `select` operation is a hard failure.
- Seeds are fixed in `q1_hypotheses.yaml`; failed runs remain in the run ledger.

## Primary analyses

1. H1 uses paired object-level fidelity for `Base` and `FuzzyXAI+Base` under the same model, object, background, feature budget, seed and time budget. Non-inferiority is accepted only when the lower 95% confidence bound is at least `-0.02`.
2. H2 measures complete provenance paths and controlled removal of required channels. False certification is reported with precision, recall, F1 and source localization.
3. H3 compares threshold-only, always-full, matched-coverage random, explainer-disagreement and adaptive A/B/C policies. Costs are frozen for safety-heavy, balanced and review-expensive regimes.
4. H4 compares always `F0`, `Fint`, `NAS`, `FML`, adaptive and diagnostic refusal. The primary comparison is adaptive versus always-FML with risk margin `0.02`.
5. H5 separates structural route diagnosis (CR-S) from incremental prediction of wrong automatic decisions (CR-P). No predictive wording is allowed without positive held-out incremental value.
6. H6 uses at least 50 paired folds/seeds and a matched random rule. Conditional effects are exploratory unless confirmed on a separate partition.
7. H7 remains an external gate until participant-level records exist and pass integrity checks.

## Statistics

Reports include effect size, 95% confidence interval, number and unit of observations, paired bootstrap or hierarchical bootstrap as applicable, a paired test, and Holm correction. P-values are never interpreted without effect sizes and intervals. Power calculations are generated before the full run.

## Missing or failed evidence

Unavailable libraries, failed explainers, incomplete downloads, NaN metrics and missing reviews are preserved as explicit statuses. They cannot be replaced by a proxy under the original method name and cannot produce a `supported` claim.

## Claim boundary

Controlled perturbations validate detector behavior, not external-domain generalization. Medical datasets are methodological research benchmarks only. Attribution methods are associational. Human usefulness and domain-language approval require real independent participants.
