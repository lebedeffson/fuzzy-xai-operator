# Full empirical validation

The E1-E8 package validates controlled multimodal behavior, repeated rule
ablation, explainer baselines, decision policies, calibration, sensitivity,
uncertainty selection, critical rupture, and scaling.

## Commands

Small local check, constrained to one numerical thread:

```bash
make empirical-smoke
```

Full release-candidate reproduction:

```bash
make reproduce-dissertation
```

Container reproduction:

```bash
docker compose run --rm reproduce
```

## Evidence policy

- Missing optional libraries or reports block their gate.
- Controlled 10,000-object generators test protocol behavior but do not establish external validity.
- Failed or inconclusive comparisons remain in the package.
- If adaptive selection uses `FML` for more than 90% of objects, the practical hierarchy claim is blocked.
- Critical rupture is called a safety indicator only if it adds measured predictive value over simple baselines.
- External comprehension and domain reviews remain `planned_not_run` or `pending_external_review` until real responses exist.
