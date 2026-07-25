# H9 E2E v2 Decision Rules

The original H9 E2E result remains `H9_E2E_TARGET_NOT_MET`.

Online mode passes its absolute target only when median overhead is at most
0.10 ms/object and p95 overhead is at most 0.50 ms/object.

Relative overhead is classified only for pipeline/batch groups whose median
model plus explainer time is at least 1 ms. Faster groups remain descriptive
and cannot fail or pass the relative gate through division by a near-zero
baseline.

Full archival export is reported separately and is not part of the online
absolute gate. Negative results are retained without threshold changes.
