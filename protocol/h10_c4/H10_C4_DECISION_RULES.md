# H10-C4 Decision Rules

## Scope

H10-C4 is a prospective algorithmic experiment on controlled structural
mutations. It does not evaluate natural software incidents, human
comprehension, engineer time, organizational cost, or production safety.
H10-C3 R4 remains an immutable parent result and is not recomputed.

## Primary comparison

The unit of analysis is `scenario_id`. The primary endpoint is normalized
executable cost. Each strategy starts from the same source snapshot and receives
the same contracts, allowed repair actions, and verifier.

The three registered paired contrasts are:

1. `O_GLOBAL - B_ALL`
2. `O_GLOBAL - B_FIRST`
3. `O_GLOBAL - B_GREEDY`

A negative difference favors `O_GLOBAL`. Confidence intervals use a paired
bootstrap with 10,000 common-index iterations. Primary p-values are corrected
with Holm's method.

## Supported result

`H10_C4_SUPPORTED` requires all of the following:

- `O_GLOBAL` repair success is not lower than any baseline;
- `O_GLOBAL` creates no new critical violations;
- every primary paired cost difference favors `O_GLOBAL`;
- every primary 95% confidence interval excludes zero;
- at least one registered secondary endpoint improves against the strongest
  successful baseline.

The registered secondary endpoints are repair action count, touched component
count, recertification check count, and measured execution time.

If structural optimality is reproduced but no registered operational endpoint
improves, the result is:

```text
STRUCTURAL_OPTIMALITY_CONFIRMED
OPERATIONAL_ADVANTAGE_NOT_CONFIRMED
```

No result may be reclassified manually after execution.
