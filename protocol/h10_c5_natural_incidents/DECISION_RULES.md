# H10-C5 Decision Rules

The method receives the buggy commit identifier, issue text, failing-test
identifiers, dependency/configuration metadata and observable logs. It never
receives the gold patch or its changed-file list.

Primary success requires both the gold source component and contract family.
Abstention is a primary failure and is also reported separately. Comparisons
are `O_ROUTE - B_TRACE`, `O_ROUTE - B_RULE`, and `O_ROUTE - B_GREEDY`, using
one shared 10,000-iteration repository-cluster bootstrap stream and Holm
correction.

`H10_C5_SUPPORTED` requires the locked positive rule. Fewer than 20 accepted
incidents yields `H10_C5_BLOCKED_DATA`. Missing independent project execution
is reported and prevents claims of executable natural-incident recovery; it
does not invalidate localization metrics derived from benchmark-verified gold.
