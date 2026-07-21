# Full empirical validation report

## Git
- branch: `feat/full-empirical-validation`
- commit: `fc8705ececdcc58f6cc171017984d9c6af05dd45`
- profile: `full`
- release status: `BLOCKED`

## E1-E8

| Experiment | Technical status | Evidence origin |
|---|---|---|
| E1 | PASS | controlled |
| E2 | PASS | controlled |
| E3 | PASS | controlled |
| E4 | PASS | controlled |
| E5 | PASS | controlled |
| E6 | PASS | controlled |
| E7 | PASS | controlled |
| E8 | PASS | controlled |

## Measured conclusions
- Rule ablation: the rule-removal effect is not confirmed as a general pattern and remains a controlled illustration.
- Required explainer methods measured: `True`.
- Adaptive FML selection fraction: `0.552200`.
- Hierarchy utility claim allowed: `True`.
- Critical-rupture incremental AUPRC: `-0.125560`.
- Critical-rupture safety claim allowed: `False`.

## External gates
- `comprehension_pilot`: `planned_not_run`
- `domain_semantic_review`: `pending_external_review`
- `expert_review`: `planned_not_run`

## Forbidden conclusions
- no clinical safety claim;
- no universal superiority claim;
- no external-domain generalization from controlled datasets;
- no stable release while external gates are incomplete.
