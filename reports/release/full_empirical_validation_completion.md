# FuzzyXAI full empirical validation completion

## Technical closure

- branch: `feat/full-empirical-validation`;
- implementation commit: `fc8705ececdcc58f6cc171017984d9c6af05dd45`;
- public workflow: `29707225733`;
- protocol smoke: PASS;
- full E1-E8: PASS;
- Docker reproduction: PASS;
- aggregate evidence: PASS;
- technical Definition of Done: `90/90 PASS`;
- stable release: BLOCKED.

## Measured scope

- four controlled modality datasets with 10,000 objects each;
- 50 paired rule-ablation comparisons over 10 folds and 5 seeds;
- SHAP, LIME, Anchors, RuleFit, confidence and declared FuzzyXAI baselines measured;
- P1-P7 policy comparison with validation-only calibration;
- sensitivity, adaptive uncertainty hierarchy and 1k/5k/10k/50k scalability measured;
- automatically generated Chapter 3-4 tables, figures and claim provenance.

## Negative results retained

- rule removal did not confirm a general subgroup-recall effect;
- critical-rupture incremental AUPRC is `-0.12556024709557384`;
- the critical rupture is therefore a structural diagnostic indicator only;
- no safety, universal superiority, clinical validity or external-domain generalization claim is allowed.

## External gates

- expert review: `planned_not_run`;
- comprehension pilot: `planned_not_run`;
- domain semantic review: `pending_external_review`;
- merge to `main`: blocked;
- stable tag: forbidden.

## Release artifacts

| Artifact | SHA256 |
|---|---|
| `fuzzyxai-full-empirical-evidence-fc8705ececdc.zip` | `244f610d26d0c2c895e26eb9c5830f4477c4772d9a0e64c737e23291ebc32c99` |
| `fuzzyxai-dissertation-artifacts-fc8705ececdc.zip` | `f608c8f1ad814e07278d0376f96c1312dc8f656a3ace0620bf62a892755aacfc` |
| `fuzzyxai-reproducibility-bundle-fc8705ececdc.zip` | `9e9acc72eb42f869c8b6cbf0ced9b26b8d9afb28333eae0e94078dec9f6a2764` |

The archive verification report is `release_artifacts/empirical_archive_verification.json`.
