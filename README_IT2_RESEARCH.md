# FuzzyXAI interval type-2 exploratory extension

This implementation uses interval type-2 uncertainty in the **model inference**
path: paired low/high memberships form exact bounds on an additive fuzzy logit,
with a sigmoid transforming the two end points. A nominal Type-1 score and an
explicit end-point midpoint are emitted as separate outputs. Thus the IT2
interval can affect both a point prediction and the abstention policy.

The footprint-of-uncertainty rectangle is estimated from train-only bootstrap
quantiles. This interval is NOT calibrated to contain the true diagnosis or
future prediction. No interval type-3 implementation is claimed.

## Branch and test status

The target existing branch is `chapter6-ophthalmology`, observed at
`8e45ba8ca9ab2e1217e12d98a3ec9a06d7cd413a`. The remote GitHub connector
rejects writes with 403; this ZIP is a **local overlay**, not a pushed branch.
See `BRANCH_INTEGRATION.md` for exact commands.

### Validated locally

- Type-2 membership nesting, coupled antecedents, exact logit extrema.
- Independent replay of a sample receipt against an externally supplied model
  fingerprint; altered outputs, manifest data, and identities are rejected.
- Boundary intersection, excessive uncertainty width and missing source guards.
- Five train/validation/test WDBC splits; matched Type-1 abstention comparator;
  24 train-only bootstrap resamples per split. See evidence ZIP.
- 108 local isolated tests across the v5 + IT2 components.

### Not yet validated

- Full original repository integration and CI in `chapter6-ophthalmology`.
- Actual PTB-XL recordings or ophthalmology model integration.
- Any clinical safety or predictive interval coverage.
- Independent semantic evaluation of explanatory claims.

## Reproduce

After applying overlay and running the transactional upgrade installer in the
full checked-out repository, run:

```bash
python -m pytest -q tests/test_doctoral_interval_type2.py
python -m experiments.doctoral_interval_type2 --out reports/chapter5/interval_type2 --bootstrap 24
```

`interval_type2.py` is purposely self-contained for independent receipt replay.
The documented Karnik-Mendel center-of-sets helper belongs to normalized
singleton-consequent inference and is **not** used to compute the bounds of
this additive fuzzy-logistic model.
