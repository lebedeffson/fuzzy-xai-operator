# H10-C5b Decision Rules

The prior H10-C5 result remains `H10_C5_NOT_SUPPORTED`.

H10-C5b is supported only when all locked conditions hold:

1. repository-cluster bootstrap CI lower bound for O_ROUTE versus B_GREEDY
   on joint file/symbol/contract Hit@3 is greater than zero;
2. O_ROUTE coverage is at least 0.70;
3. O_ROUTE false localization is not greater than B_GREEDY;
4. at least eight held-out repositories and 24 held-out incidents are scored;
5. Gold leakage is zero.

Recovery endpoints are evaluated only for incidents whose repair is expressible
by a preregistered dependency, configuration, schema, serialization, artifact,
or version operation. A localization-only incident cannot contribute a
successful recovery.

No sealed scoring is allowed until the structural unit scenarios, development
repository calibration, source commitments, method lock, and leakage audit
pass.
