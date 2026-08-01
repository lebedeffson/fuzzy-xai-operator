# H10-C5 Inclusion and Exclusion

Include an incident only when the source benchmark supplies a repository,
buggy commit, issue description, at least one failing-to-passing test, and a
gold patch. The incident must map to one of the locked route-contract families:

`DEPENDENCY_VERSION`, `MODEL_EXPLAINER_VERSION`, `PREPROCESSING_SCHEMA`,
`DATA_CONTRACT`, `ARTIFACT_PROVENANCE`, `SERIALIZATION`,
`CHECKSUM_INTEGRITY`, `PIPELINE_CONFIGURATION`, `MODEL_LOADING`, or
`EXPLAINER_CONFIGURATION`.

Exclude UI-only and documentation-only changes, feature requests, pure
performance regressions, arbitrary algorithm defects, incidents without a
reproduction test, and cases whose patch cannot be mapped to a formal route
component without subjective adjudication.

The screening algorithm and per-repository quotas are frozen before held-out
scoring. A benchmark-provided verified test is recorded separately from a
locally executed project test.
