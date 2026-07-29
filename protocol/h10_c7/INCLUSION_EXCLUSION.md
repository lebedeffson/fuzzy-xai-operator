# H10-C7 inclusion and exclusion

## Development

- At least 40 incidents from at least 10 repositories.
- H10-C5b and H10-C5c data are open development data only.
- Every incident must have a reproduced failing test, project-grounded
  traceback and a repository graph built without the future patch.
- Gold is supplied to the scorer in a separate file after all predictions.

## Confirmatory held-out

- At least 40 incidents from at least 12 repository-disjoint repositories.
- At least four registered contract families and complete runtime evidence.
- Every H10-C5b and H10-C5c development or held-out repository is excluded.
- The fix commit, patch, changed files, changed symbols and post-fix maintainer
  explanation remain sealed until the one-time score.
- Held-out creation is forbidden unless every development gate passes and a
  method lock exists.

The preferred source is a commit-pinned SWE-bench-Live snapshot. Availability
does not override repository disjointness or runtime reproducibility.
